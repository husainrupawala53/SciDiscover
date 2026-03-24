"""
db/neo4j_db.py — Neo4j driver wrapper with async-compatible helpers.
Uses the official neo4j-driver (sync sessions run in thread pools).
"""

from neo4j import GraphDatabase, Driver
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)

_driver: Driver = None


def connect_neo4j() -> None:
    """Initialize the Neo4j bolt driver."""
    global _driver
    settings = get_settings()
    _driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    _driver.verify_connectivity()
    logger.info("✅ Neo4j connected successfully")


def close_neo4j() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        logger.info("Neo4j connection closed")


def get_neo4j_driver() -> Driver:
    """Return the shared Neo4j driver instance."""
    return _driver


def run_query(cypher: str, parameters: dict = None):
    """
    Execute a Cypher query in a managed session.
    Returns a list of record dicts.
    """
    with _driver.session() as session:
        result = session.run(cypher, parameters or {})
        return [record.data() for record in result]
