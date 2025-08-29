#!/usr/bin/env python3
"""
Image URL Validator with concurrent processing.

This script:
1. Finds all urls.txt files in the test-images directory
2. Validates URLs concurrently using asyncio and aiohttp
3. Checks if URLs point to valid images
4. Optionally downloads images
5. Provides a comprehensive report

Performance improvements:
- Concurrent HTTP requests using asyncio
- Connection pooling for reduced overhead
- Configurable concurrency limits
- Batch processing of files
"""

import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Tuple, Optional
import time
import json
import aiohttp
import aiofiles
from dataclasses import dataclass, asdict
import typer


@dataclass
class URLResult:
    """Result of URL validation."""
    url: str
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    is_image: bool = False
    error: Optional[str] = None
    response_time: float = 0.0


class ImageURLValidator:
    """SmartRobe Image URL validator using concurrent HTTP requests with rate limiting."""
    
    def __init__(self, test_images_dir: Path, download_dir: Optional[Path] = None, 
                 max_concurrent: int = 10, timeout: int = 10, request_delay: float = 0.1):
        self.test_images_dir = Path(test_images_dir)
        self.download_dir = Path(download_dir) if download_dir else None
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.request_delay = request_delay  # Delay between requests to prevent rate limiting
        self.results = {}
        self.download_errors = []  # Track download errors separately
        self.stats = {
            'total_files': 0,
            'total_urls': 0,
            'valid_urls': 0,
            'invalid_urls': 0,
            'image_urls': 0,
            'non_image_urls': 0,
            'downloaded': 0,
            'download_errors': 0,
            'download_retries': 0,
            'total_time': 0.0,
            'avg_response_time': 0.0
        }
    
    def find_urls_files(self) -> List[Path]:
        """Find all urls.txt files in the test-images directory."""
        urls_files = []
        for root, dirs, files in os.walk(self.test_images_dir):
            if 'urls.txt' in files:
                urls_files.append(Path(root) / 'urls.txt')
        return sorted(urls_files)
    
    async def read_urls_from_file(self, file_path: Path) -> List[str]:
        """Read URLs from a urls.txt file asynchronously."""
        urls = []
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    url = line.strip()
                    if url and not url.startswith('#'):  # Skip empty lines and comments
                        urls.append(url)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return urls
    
    async def validate_url(self, session: aiohttp.ClientSession, url: str) -> URLResult:
        """
        Validate a URL using async HTTP request with rate limiting.
        
        Returns:
            URLResult with validation information
        """
        result = URLResult(url=url)
        start_time = time.time()
        
        try:
            # Add small delay to prevent rate limiting
            await asyncio.sleep(self.request_delay)
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; SmartRobe-Image-URL-Validator/2.0)'
            }
            
            # Use HEAD request first for efficiency
            async with session.head(url, timeout=timeout, headers=headers) as response:
                result.status_code = response.status
                result.content_type = response.headers.get('Content-Type', '')
                content_length = response.headers.get('Content-Length')
                if content_length:
                    try:
                        result.content_length = int(content_length)
                    except ValueError:
                        pass
                
                # Check if it's an image
                content_type = result.content_type.lower()
                result.is_image = any(img_type in content_type for img_type in 
                                     ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp', 'bmp'])
                
                result.response_time = time.time() - start_time
                return result
                
        except asyncio.TimeoutError:
            result.error = f"Timeout after {self.timeout}s"
        except aiohttp.ClientError as e:
            result.error = f"Client error: {str(e)}"
        except Exception as e:
            result.error = f"Unexpected error: {str(e)}"
        
        result.response_time = time.time() - start_time
        return result
    
    async def download_image(self, session: aiohttp.ClientSession, url: str, 
                           download_path: Path, max_retries: int = 3) -> bool:
        """Download an image from URL to the specified path asynchronously with retry logic."""
        for attempt in range(max_retries):
            try:
                # Add delay to prevent rate limiting
                if attempt > 0:
                    delay = self.request_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    self.stats['download_retries'] += 1
                
                timeout = aiohttp.ClientTimeout(total=30)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; SmartRobe-Image-Downloader/2.0)'
                }
                
                async with session.get(url, timeout=timeout, headers=headers) as response:
                    if response.status == 200:
                        download_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        async with aiofiles.open(download_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        return True
                    elif response.status in [429, 503, 502]:  # Rate limited or server error
                        error_msg = f"Download attempt {attempt + 1} failed for {url}: HTTP {response.status}"
                        self.download_errors.append(error_msg)
                        if attempt == max_retries - 1:
                            return False
                        continue
                    else:
                        error_msg = f"Download failed for {url}: HTTP {response.status}"
                        self.download_errors.append(error_msg)
                        return False
                        
            except asyncio.TimeoutError:
                error_msg = f"Download timeout (attempt {attempt + 1}) for {url}"
                self.download_errors.append(error_msg)
                if attempt == max_retries - 1:
                    return False
            except Exception as e:
                error_msg = f"Download error (attempt {attempt + 1}) for {url}: {e}"
                self.download_errors.append(error_msg)
                if attempt == max_retries - 1:
                    return False
        
        return False
    
    def get_filename_from_url(self, url: str) -> str:
        """Extract unique filename from URL, preventing collisions."""
        import hashlib
        
        parsed = urlparse(url)
        original_filename = Path(parsed.path).name
        
        # Create a unique identifier from the full URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if original_filename and '.' in original_filename:
            # Use original filename but make it unique with hash
            name, ext = original_filename.rsplit('.', 1)
            filename = f"{name}_{url_hash}.{ext}"
        else:
            # No proper filename found, generate one
            filename = f"image_{url_hash}.jpg"
        
        return filename
    
    async def process_file(self, session: aiohttp.ClientSession, urls_file: Path, 
                          download: bool = False) -> Tuple[Path, List[URLResult]]:
        """Process a single urls.txt file."""
        relative_path = urls_file.relative_to(self.test_images_dir)
        urls = await self.read_urls_from_file(urls_file)
        
        if not urls:
            return relative_path, []
        
        print(f"Processing: {relative_path} ({len(urls)} URLs)")
        
        # Create semaphore to limit concurrent requests per file
        semaphore = asyncio.Semaphore(min(self.max_concurrent, len(urls)))
        
        async def validate_with_semaphore(url: str) -> URLResult:
            async with semaphore:
                return await self.validate_url(session, url)
        
        # Validate all URLs concurrently
        tasks = [validate_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = URLResult(url=urls[i], error=str(result))
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        # Download images if requested
        if download and self.download_dir:
            download_tasks = []
            for result in valid_results:
                if result.is_image and not result.error:
                    filename = self.get_filename_from_url(result.url)
                    download_path = self.download_dir / relative_path.parent / filename
                    download_tasks.append(
                        self.download_image(session, result.url, download_path)
                    )
            
            if download_tasks:
                download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
                successful_downloads = sum(1 for r in download_results if r is True)
                failed_downloads = len(download_results) - successful_downloads
                self.stats['downloaded'] += successful_downloads
                self.stats['download_errors'] += failed_downloads
        
        return relative_path, valid_results
    
    async def validate_all_urls(self, download: bool = False) -> None:
        """Validate all URLs found in urls.txt files concurrently."""
        start_time = time.time()
        
        urls_files = self.find_urls_files()
        self.stats['total_files'] = len(urls_files)
        
        print(f"Found {len(urls_files)} urls.txt files")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Timeout: {self.timeout}s")
        print("=" * 60)
        
        # Configure aiohttp session with conservative connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,      # Total connection pool size
            limit_per_host=3,               # Max connections per host (reduced)
            enable_cleanup_closed=True,
            ttl_dns_cache=300,              # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=10,           # Keep connections alive for reuse
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; SmartRobe-Image-URL-Validator/2.0)'}
        ) as session:
            # Process all files concurrently with limited concurrency
            file_semaphore = asyncio.Semaphore(min(5, len(urls_files)))  # Max 5 files at once
            
            async def process_with_semaphore(urls_file):
                async with file_semaphore:
                    return await self.process_file(session, urls_file, download)
            
            tasks = [process_with_semaphore(urls_file) for urls_file in urls_files]
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results and statistics
            total_response_time = 0.0
            valid_responses = 0
            
            for file_result in file_results:
                if isinstance(file_result, Exception):
                    print(f"Error processing file: {file_result}")
                    continue
                
                relative_path, results = file_result
                self.results[str(relative_path)] = [asdict(r) for r in results]
                
                for result in results:
                    self.stats['total_urls'] += 1
                    
                    if result.error:
                        self.stats['invalid_urls'] += 1
                        status = f"✗ Invalid - {result.error}"
                    else:
                        self.stats['valid_urls'] += 1
                        valid_responses += 1
                        total_response_time += result.response_time
                        
                        if result.is_image:
                            self.stats['image_urls'] += 1
                            status = f"✓ Valid image ({result.content_type}"
                            if result.content_length:
                                status += f", {result.content_length:,} bytes"
                            status += f", {result.response_time:.2f}s)"
                        else:
                            self.stats['non_image_urls'] += 1
                            status = f"⚠ Valid but not an image ({result.content_type}, {result.response_time:.2f}s)"
                    
                    # Show a subset of results to avoid overwhelming output
                    if self.stats['total_urls'] <= 20 or self.stats['total_urls'] % 20 == 0:
                        print(f"  {result.url[:60]}{'...' if len(result.url) > 60 else ''}")
                        print(f"    {status}")
        
        self.stats['total_time'] = time.time() - start_time
        if valid_responses > 0:
            self.stats['avg_response_time'] = total_response_time / valid_responses
    
    def print_summary(self) -> None:
        """Print a summary of validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Files processed: {self.stats['total_files']}")
        print(f"Total URLs: {self.stats['total_urls']}")
        print(f"Valid URLs: {self.stats['valid_urls']}")
        print(f"Invalid URLs: {self.stats['invalid_urls']}")
        print(f"Image URLs: {self.stats['image_urls']}")
        print(f"Non-image URLs: {self.stats['non_image_urls']}")
        
        if self.download_dir:
            print(f"Successfully downloaded: {self.stats['downloaded']}")
            print(f"Download errors: {self.stats['download_errors']}")
            print(f"Download retries: {self.stats['download_retries']}")
        
        print(f"\nPerformance:")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"URLs per second: {self.stats['total_urls'] / self.stats['total_time']:.1f}")
        print(f"Average response time: {self.stats['avg_response_time']:.3f}s")
        
        if self.stats['total_urls'] > 0:
            valid_percent = (self.stats['valid_urls'] / self.stats['total_urls']) * 100
            image_percent = (self.stats['image_urls'] / self.stats['total_urls']) * 100
            print(f"\nSuccess rate: {valid_percent:.1f}% valid URLs")
            print(f"Image rate: {image_percent:.1f}% are images")
        
        # Show recent download errors
        if self.download_errors:
            print(f"\nRecent Download Errors (showing last 10):")
            for error in self.download_errors[-10:]:
                print(f"  • {error}")
    
    async def save_detailed_report(self, output_file: Path) -> None:
        """Save a detailed JSON report of all results asynchronously."""
        report = {
            'stats': self.stats,
            'results': self.results,
            'download_errors': self.download_errors,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))
        
        print(f"\nDetailed report saved to: {output_file}")


async def main(
    test_images_dir: Path = typer.Option(
        Path("test-images"),
        "--test-images-dir",
        help="Path to test-images directory"
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Download valid images"
    ),
    download_dir: Path = typer.Option(
        Path("downloaded-images"),
        "--download-dir",
        help="Directory to download images to"
    ),
    save_report: bool = typer.Option(
        True,
        "--save-report/--no-save-report",
        help="Save detailed JSON report"
    ),
    report: Path = typer.Option(
        Path("url_validation_report.json"),
        "--report",
        help="Path to save detailed JSON report"
    ),
    max_concurrent: int = typer.Option(
        10,
        "--max-concurrent",
        help="Maximum concurrent requests"
    ),
    request_delay: float = typer.Option(
        0.1,
        "--request-delay",
        help="Delay between requests in seconds (helps prevent rate limiting)"
    ),
    timeout: int = typer.Option(
        10,
        "--timeout",
        help="Request timeout in seconds"
    )
) -> None:
    """SmartRobe Image URL Validator with concurrent processing."""
    
    # Validate input directory
    if not test_images_dir.exists():
        typer.echo(f"Error: Directory {test_images_dir} does not exist")
        raise typer.Exit(1)
    
    # Initialize validator
    download_dir_final = download_dir if download else None
    validator = ImageURLValidator(
        test_images_dir, 
        download_dir_final,
        max_concurrent=max_concurrent,
        timeout=timeout,
        request_delay=request_delay
    )
    
    typer.echo(f"SmartRobe Image URL Validator")
    typer.echo(f"Test images directory: {test_images_dir.absolute()}")
    if download:
        typer.echo(f"Download directory: {download_dir_final.absolute()}")
    if save_report:
        typer.echo(f"Report will be saved to: {report.absolute()}")
    typer.echo()
    
    try:
        # Validate all URLs
        await validator.validate_all_urls(download=download)
        
        # Print summary
        validator.print_summary()
        
        # Save detailed report if requested
        if save_report:
            await validator.save_detailed_report(report)
        
    except KeyboardInterrupt:
        typer.echo("\n\nValidation interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\nUnexpected error: {e}")
        raise typer.Exit(1)


def cli_main(
    test_images_dir: Path = typer.Option(
        Path("test-images"),
        "--test-images-dir",
        help="Path to test-images directory"
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Download valid images"
    ),
    download_dir: Path = typer.Option(
        Path("downloaded-images"),
        "--download-dir",
        help="Directory to download images to"
    ),
    save_report: bool = typer.Option(
        True,
        "--save-report/--no-save-report",
        help="Save detailed JSON report"
    ),
    report: Path = typer.Option(
        Path("url_validation_report.json"),
        "--report",
        help="Path to save detailed JSON report"
    ),
    max_concurrent: int = typer.Option(
        10,
        "--max-concurrent",
        help="Maximum concurrent requests"
    ),
    request_delay: float = typer.Option(
        0.1,
        "--request-delay",
        help="Delay between requests in seconds (helps prevent rate limiting)"
    ),
    timeout: int = typer.Option(
        10,
        "--timeout",
        help="Request timeout in seconds"
    )
) -> None:
    """SmartRobe Image URL Validator with concurrent processing."""
    asyncio.run(main(
        test_images_dir=test_images_dir,
        download=download,
        download_dir=download_dir,
        save_report=save_report,
        report=report,
        max_concurrent=max_concurrent,
        request_delay=request_delay,
        timeout=timeout
    ))


if __name__ == "__main__":
    typer.run(cli_main)
