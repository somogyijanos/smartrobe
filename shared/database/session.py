"""
Database session management for Smartrobe services.

Provides async database session handling with connection pooling,
health checks, and proper error handling for PostgreSQL.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from .models import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
            echo: Whether to echo SQL statements (for debugging)
        """
        self.database_url = database_url
        self.echo = echo
        self._engine = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return

        logger.info("Initializing database connection", url=self.database_url)

        # Create async engine with connection pooling
        self._engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._initialized = True
        logger.info("Database connection initialized successfully")

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized:
            await self.initialize()

        logger.info("Creating database tables")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    async def drop_tables(self) -> None:
        """Drop all database tables (for testing/development)."""
        if not self._initialized:
            await self.initialize()

        logger.warning("Dropping all database tables")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession: Database session with automatic cleanup
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()

            async with self.get_session() as session:
                # Simple query to check connectivity
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                return True

        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_database_manager(
    database_url: str | None = None, echo: bool = False
) -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Args:
        database_url: Database URL (required on first call)
        echo: Echo SQL statements
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        if database_url is None:
            raise ValueError("database_url is required for first call")
        _db_manager = DatabaseManager(database_url, echo)
    
    return _db_manager


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get a database session.
    
    Yields:
        AsyncSession: Database session
    """
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    
    async with _db_manager.get_session() as session:
        yield session


async def init_database(
    database_url: str, create_tables: bool = True, echo: bool = False
) -> None:
    """
    Initialize database and optionally create tables.
    
    Args:
        database_url: PostgreSQL connection URL
        create_tables: Whether to create tables
        echo: Echo SQL statements
    """
    db_manager = get_database_manager(database_url, echo)
    await db_manager.initialize()
    
    if create_tables:
        await db_manager.create_tables()


async def close_database() -> None:
    """Close database connections."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None
