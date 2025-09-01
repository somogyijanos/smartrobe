#!/usr/bin/env python3
"""
Simple testing script for SmartRobe /analyze endpoint.

Usage examples:
    python scripts/test_analyze.py 01
    python scripts/test_analyze.py 23 --endpoint http://localhost:8000
    python scripts/test_analyze.py 05 --save-response results.json
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
import time

import aiohttp
import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

async def read_urls_from_file(file_path: Path) -> list[str]:
    """Read URLs from urls.txt file."""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):
                    urls.append(url)
    except Exception as e:
        console.print(f"[red]Error reading {file_path}: {e}[/red]")
        return []
    return urls

async def make_analyze_request(endpoint: str, urls: list[str]) -> tuple[Optional[dict], Optional[str]]:
    """Make the /analyze request and return response or error."""
    analyze_url = f"{endpoint}/v1/items/analyze"
    payload = {"images": urls}
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {'Content-Type': 'application/json'}
            
            console.print(f"[yellow]Making request to: {analyze_url}[/yellow]")
            console.print(f"[yellow]With {len(urls)} image URLs[/yellow]")
            
            start_time = time.time()
            
            async with session.post(analyze_url, json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    console.print(f"[green]✓ Success! Response time: {response_time:.2f}s[/green]")
                    return result, None
                else:
                    error_text = await response.text()
                    error = f"HTTP {response.status}: {error_text}"
                    console.print(f"[red]✗ Request failed: {error}[/red]")
                    return None, error
                    
    except asyncio.TimeoutError:
        error = "Request timed out after 60 seconds"
        console.print(f"[red]✗ {error}[/red]")
        return None, error
    except Exception as e:
        error = f"Request error: {str(e)}"
        console.print(f"[red]✗ {error}[/red]")
        return None, error

def display_response(response: dict) -> None:
    """Display the analyze response in a nice format."""
    
    # Response overview
    processing_info = response.get('processing', {})
    console.print(Panel(
        f"[bold]Request ID:[/bold] {response.get('id', 'N/A')}\n"
        f"[bold]Processing Time:[/bold] {processing_info.get('total_processing_time_ms', 'N/A')}ms\n"
        f"[bold]Images Processed:[/bold] {processing_info.get('image_count', 'N/A')}\n"
        f"[bold]Timestamp:[/bold] {processing_info.get('timestamp', 'N/A')}",
        title="[bold blue]Analysis Overview[/bold blue]",
        expand=False
    ))
    
    # Extracted attributes
    attributes = response.get('attributes', {})
    if attributes:
        attr_table = Table(title="[bold green]Extracted Attributes[/bold green]")
        attr_table.add_column("Attribute", style="cyan", no_wrap=True)
        attr_table.add_column("Value", style="magenta")
        
        for key, value in attributes.items():
            # Skip None values
            if value is not None:
                attr_table.add_row(key, str(value))
        
        console.print(attr_table)
    
    # Model information
    model_info = response.get('model_info', {})
    if model_info:
        console.print("\n[bold yellow]Model Performance:[/bold yellow]")
        
        for model_name, model_data in model_info.items():
            if isinstance(model_data, dict):
                processing_time = model_data.get('processing_time_ms', 'N/A')
                success = model_data.get('success', False)
                status = "[green]✓[/green]" if success else "[red]✗[/red]"
                
                console.print(f"  {status} [bold]{model_name}[/bold]: {processing_time}ms")
                
                # Show confidence scores if available
                confidence_scores = model_data.get('confidence_scores', {})
                if confidence_scores:
                    for attr, score in confidence_scores.items():
                        if score is not None:
                            console.print(f"    • {attr}: {score:.3f}")

async def main(
    test_number: str = typer.Argument(
        ...,
        help="Test image directory number (e.g., 01, 02, 23)"
    ),
    endpoint: str = typer.Option(
        "http://localhost:8000",
        "--endpoint",
        help="SmartRobe API endpoint"
    ),
    test_images_dir: Path = typer.Option(
        Path("test-images"),
        "--test-images-dir",
        help="Path to test-images directory"
    ),
    save_response: Optional[Path] = typer.Option(
        None,
        "--save-response",
        help="Save full response to JSON file"
    ),
    show_raw: bool = typer.Option(
        False,
        "--raw",
        help="Show raw JSON response"
    )
) -> None:
    """Test SmartRobe /analyze endpoint with images from test directories."""
    
    # Validate test number format
    test_number = test_number.zfill(2)  # Pad with zero if needed
    
    # Find the urls.txt file
    urls_file = test_images_dir / test_number / "urls.txt"
    
    if not urls_file.exists():
        console.print(f"[red]Error: File {urls_file} does not exist[/red]")
        console.print(f"[yellow]Available test directories:[/yellow]")
        if test_images_dir.exists():
            for item in sorted(test_images_dir.iterdir()):
                if item.is_dir() and (item / "urls.txt").exists():
                    console.print(f"  • {item.name}")
        raise typer.Exit(1)
    
    # Read URLs
    console.print(f"[blue]Reading URLs from: {urls_file}[/blue]")
    urls = await read_urls_from_file(urls_file)
    
    if not urls:
        console.print(f"[red]No valid URLs found in {urls_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Found {len(urls)} image URLs[/blue]")
    for i, url in enumerate(urls, 1):
        console.print(f"  {i}. {url}")
    
    console.print()
    
    # Make the request
    response, error = await make_analyze_request(endpoint, urls)
    
    if error:
        console.print(f"\n[red]Failed to get response: {error}[/red]")
        raise typer.Exit(1)
    
    if not response:
        console.print(f"\n[red]No response received[/red]")
        raise typer.Exit(1)
    
    console.print()
    
    # Display response
    if show_raw:
        console.print(Panel(
            JSON.from_data(response),
            title="[bold]Raw JSON Response[/bold]",
            expand=False
        ))
    else:
        display_response(response)
    
    # Save response if requested
    if save_response:
        try:
            with open(save_response, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Response saved to: {save_response}[/green]")
        except Exception as e:
            console.print(f"\n[red]Failed to save response: {e}[/red]")

def cli_main(
    test_number: str = typer.Argument(
        ...,
        help="Test image directory number (e.g., 01, 02, 23)"
    ),
    endpoint: str = typer.Option(
        "http://localhost:8000",
        "--endpoint",
        help="SmartRobe API endpoint"
    ),
    test_images_dir: Path = typer.Option(
        Path("test-images"),
        "--test-images-dir",
        help="Path to test-images directory"
    ),
    save_response: Optional[Path] = typer.Option(
        None,
        "--save-response",
        help="Save full response to JSON file"
    ),
    show_raw: bool = typer.Option(
        False,
        "--raw",
        help="Show raw JSON response"
    )
) -> None:
    """Test SmartRobe /analyze endpoint with images from test directories."""
    
    try:
        asyncio.run(main(
            test_number=test_number,
            endpoint=endpoint,
            test_images_dir=test_images_dir,
            save_response=save_response,
            show_raw=show_raw
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(cli_main)



