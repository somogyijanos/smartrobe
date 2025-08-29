"""
Repository pattern for database operations.

Provides high-level interface for storing and retrieving inference results
with proper error handling and logging.
"""

import uuid
from typing import Optional

from loguru import logger
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.schemas import AnalyzeResponse, InferenceResult as InferenceResultSchema

from .models import InferenceResult
from .session import get_db_session


class InferenceRepository:
    """Repository for inference result database operations."""

    @staticmethod
    async def create_inference_result(
        response: AnalyzeResponse, 
        request_data: dict
    ) -> InferenceResult:
        """
        Store a new inference result in the database.
        
        Args:
            response: The analysis response containing all extracted data
            request_data: Original request data (image URLs, etc.)
            
        Returns:
            InferenceResult: The stored database record
        """
        try:
            async with get_db_session() as session:
                # Convert response to database model
                db_result = InferenceResult(
                    id=response.id,
                    request_data=request_data,
                    attributes=response.attributes.model_dump(mode='json'),
                    model_info={
                        name: info.model_dump(mode='json') 
                        for name, info in response.model_info.items()
                    },
                    processing_info=response.processing.model_dump(mode='json'),
                )

                session.add(db_result)
                await session.flush()  # Get the ID
                await session.refresh(db_result)  # Refresh with DB values

                logger.info(
                    "Inference result stored in database",
                    result_id=str(db_result.id),
                    attribute_count=len(response.attributes.model_dump(exclude_none=True)),
                )

                return db_result

        except Exception as e:
            # Let the calling service handle the logging with more context
            raise

    @staticmethod
    async def get_inference_result(result_id: uuid.UUID) -> Optional[InferenceResult]:
        """
        Retrieve an inference result by ID.
        
        Args:
            result_id: The unique result identifier
            
        Returns:
            InferenceResult or None if not found
        """
        try:
            async with get_db_session() as session:
                stmt = select(InferenceResult).where(InferenceResult.id == result_id)
                result = await session.execute(stmt)
                db_result = result.scalar_one_or_none()

                if db_result:
                    logger.debug(
                        "Retrieved inference result from database",
                        result_id=str(result_id),
                    )
                else:
                    logger.warning(
                        "Inference result not found",
                        result_id=str(result_id),
                    )

                return db_result

        except Exception as e:
            logger.error(
                "Failed to retrieve inference result",
                result_id=str(result_id),
                error=str(e),
            )
            raise

    @staticmethod
    async def list_inference_results(
        limit: int = 100, 
        offset: int = 0
    ) -> list[InferenceResult]:
        """
        List inference results with pagination.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of InferenceResult records
        """
        try:
            async with get_db_session() as session:
                stmt = (
                    select(InferenceResult)
                    .order_by(desc(InferenceResult.created_at))
                    .limit(limit)
                    .offset(offset)
                )
                
                result = await session.execute(stmt)
                db_results = result.scalars().all()

                logger.debug(
                    "Retrieved inference results list",
                    count=len(db_results),
                    limit=limit,
                    offset=offset,
                )

                return list(db_results)

        except Exception as e:
            logger.error(
                "Failed to list inference results",
                limit=limit,
                offset=offset,
                error=str(e),
            )
            raise

    @staticmethod
    async def delete_inference_result(result_id: uuid.UUID) -> bool:
        """
        Delete an inference result by ID.
        
        Args:
            result_id: The unique result identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with get_db_session() as session:
                stmt = select(InferenceResult).where(InferenceResult.id == result_id)
                result = await session.execute(stmt)
                db_result = result.scalar_one_or_none()

                if db_result:
                    await session.delete(db_result)
                    logger.info(
                        "Inference result deleted from database",
                        result_id=str(result_id),
                    )
                    return True
                else:
                    logger.warning(
                        "Cannot delete - inference result not found",
                        result_id=str(result_id),
                    )
                    return False

        except Exception as e:
            logger.error(
                "Failed to delete inference result",
                result_id=str(result_id),
                error=str(e),
            )
            raise

    @staticmethod
    async def count_inference_results() -> int:
        """
        Count total number of inference results.
        
        Returns:
            Total count of records
        """
        try:
            async with get_db_session() as session:
                from sqlalchemy import func
                
                stmt = select(func.count(InferenceResult.id))
                result = await session.execute(stmt)
                count = result.scalar()

                logger.debug("Counted inference results", total=count)
                return count

        except Exception as e:
            logger.error("Failed to count inference results", error=str(e))
            raise
