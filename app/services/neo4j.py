from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
from neo4j.time import DateTime as Neo4jDateTime
from app.core.config import (
    NEO4J_URI, 
    NEO4J_USERNAME, 
    NEO4J_PASSWORD,
    NEO4J_DATABASE
)
import json

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self):
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            # Test the connection
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("Successfully connected to Neo4j database")
                else:
                    logger.error("Failed to validate Neo4j connection")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None

    def _convert_neo4j_values(self, value):
        """
        Convert Neo4j-specific types to Python standard types.
        This handles recursively converting values in dictionaries, lists, etc.
        """
        if isinstance(value, Neo4jDateTime):
            # Convert Neo4j DateTime to Python datetime
            return datetime.fromtimestamp(value.to_native().timestamp())
        elif isinstance(value, dict):
            # Convert all values in dictionary
            return {k: self._convert_neo4j_values(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Convert all items in list
            return [self._convert_neo4j_values(item) for item in value]
        else:
            # Return other types as is
            return value

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    # PromptGroup methods
    def create_prompt_group(self, name: str, description: str, tags: List[str] = None) -> str:
        """Create a new prompt group node in the database."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    CREATE (pg:PromptGroup {
                        id: randomUUID(),
                        name: $name,
                        description: $description,
                        tags: $tags,
                        created_at: datetime()
                    })
                    RETURN pg.id as id
                    """,
                    name=name,
                    description=description,
                    tags=tags or []
                )
                record = result.single()
                if record:
                    return record["id"]
                return None
        except Exception as e:
            logger.error(f"Error creating prompt group: {str(e)}")
            raise
    
    def get_prompt_groups(self) -> List[Dict]:
        """Get all prompt groups."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (pg:PromptGroup)
                    OPTIONAL MATCH (pg)<-[:BELONGS_TO]-(p:Prompt)
                    WITH pg, COUNT(p) as prompt_count
                    RETURN pg {
                        .*,
                        prompt_count: prompt_count
                    } as group
                    ORDER BY pg.created_at DESC
                    """
                )
                groups = [record["group"] for record in result]
                return [self._convert_neo4j_values(group) for group in groups]
        except Exception as e:
            logger.error(f"Error getting prompt groups: {str(e)}")
            raise

    def get_prompt_group(self, group_id: str) -> Dict:
        """Get a specific prompt group."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (pg:PromptGroup {id: $group_id})
                    OPTIONAL MATCH (pg)<-[:BELONGS_TO]-(p:Prompt)
                    WITH pg, COUNT(p) as prompt_count
                    RETURN pg {
                        .*,
                        prompt_count: prompt_count
                    } as group
                    """,
                    group_id=group_id
                )
                record = result.single()
                if record:
                    return self._convert_neo4j_values(record["group"])
                return None
        except Exception as e:
            logger.error(f"Error getting prompt group: {str(e)}")
            raise

    # Prompt methods
    def create_prompt(self, data) -> str:
        """Create a new prompt node in the database and link it to a prompt group."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (pg:PromptGroup {id: $prompt_group_id})
                    CREATE (p:Prompt {
                        id: randomUUID(),
                        name: $name,
                        description: $description,
                        content: $content,
                        tags: $tags,
                        created_at: datetime()
                    })
                    CREATE (p)-[:BELONGS_TO]->(pg)
                    RETURN p.id as id
                    """,
                    prompt_group_id=data.prompt_group_id,
                    name=data.name,
                    description=data.description,
                    content=data.content,
                    tags=data.tags or []
                )
                record = result.single()
                if record:
                    return record["id"]
                return None
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            raise

    def get_prompts_by_group(self, group_id: str) -> List[Dict]:
        """Get all prompts in a specific prompt group."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (p:Prompt)-[:BELONGS_TO]->(pg:PromptGroup {id: $group_id})
                    OPTIONAL MATCH (p)<-[:VERSION_OF]-(pv:PromptVersion)
                    WITH p, COUNT(pv) as version_count
                    RETURN p {
                        .*,
                        version_count: version_count
                    } as prompt
                    ORDER BY p.created_at DESC
                    """,
                    group_id=group_id
                )
                prompts = [record["prompt"] for record in result]
                return [self._convert_neo4j_values(prompt) for prompt in prompts]
        except Exception as e:
            logger.error(f"Error getting prompts by group: {str(e)}")
            raise

    def get_prompt(self, prompt_id: str) -> Dict:
        """Get a specific prompt."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (p:Prompt {id: $prompt_id})
                    OPTIONAL MATCH (p)<-[:VERSION_OF]-(pv:PromptVersion)
                    OPTIONAL MATCH (p)-[:BELONGS_TO]->(pg:PromptGroup)
                    WITH p, COUNT(pv) as version_count, pg
                    RETURN p {
                        .*,
                        version_count: version_count,
                        prompt_group_id: pg.id
                    } as prompt
                    """,
                    prompt_id=prompt_id
                )
                record = result.single()
                if record:
                    return self._convert_neo4j_values(record["prompt"])
                return None
        except Exception as e:
            logger.error(f"Error getting prompt: {str(e)}")
            raise

    # PromptVersion methods
    def create_prompt_version(self, data) -> str:
        """Create a new prompt version node and link it to its parent prompt."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (p:Prompt {id: $prompt_id})
                    CREATE (pv:PromptVersion {
                        id: randomUUID(),
                        content: $content,
                        version: $version,
                        notes: $notes,
                        created_at: datetime()
                    })
                    CREATE (pv)-[:VERSION_OF]->(p)
                    WITH pv
                    OPTIONAL MATCH (parent:PromptVersion {id: $derived_from})
                    WITH pv, parent
                    FOREACH (x IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END | 
                        CREATE (pv)-[:DERIVED_FROM]->(parent)
                    )
                    RETURN pv.id as id
                    """,
                    prompt_id=data.prompt_id,
                    content=data.content,
                    version=data.version,
                    notes=data.notes,
                    derived_from=data.derived_from
                )
                record = result.single()
                if record:
                    return record["id"]
                return None
        except Exception as e:
            logger.error(f"Error creating prompt version: {str(e)}")
            raise

    def get_prompt_versions(self, prompt_id: str) -> List[Dict]:
        """Get all versions of a specific prompt."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (pv:PromptVersion)-[:VERSION_OF]->(p:Prompt {id: $prompt_id})
                    OPTIONAL MATCH (tr:TestRun)-[:TESTED_WITH]->(pv)
                    WITH pv, COUNT(tr) as test_run_count
                    RETURN pv {
                        .*,
                        test_runs: test_run_count
                    } as version
                    ORDER BY pv.version
                    """,
                    prompt_id=prompt_id
                )
                versions = [record["version"] for record in result]
                return [self._convert_neo4j_values(version) for version in versions]
        except Exception as e:
            logger.error(f"Error getting prompt versions: {str(e)}")
            raise

    def get_prompt_version(self, version_id: str) -> Dict:
        """Get a specific prompt version."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (pv:PromptVersion {id: $version_id})
                    OPTIONAL MATCH (tr:TestRun)-[:TESTED_WITH]->(pv)
                    OPTIONAL MATCH (pv)-[:VERSION_OF]->(p:Prompt)
                    WITH pv, COUNT(tr) as test_run_count, p
                    RETURN pv {
                        .*,
                        test_runs: test_run_count,
                        prompt_id: p.id,
                        prompt_name: p.name
                    } as version
                    """,
                    version_id=version_id
                )
                record = result.single()
                if record:
                    return self._convert_neo4j_values(record["version"])
                return None
        except Exception as e:
            logger.error(f"Error getting prompt version: {str(e)}")
            raise

    # TestRun methods
    def create_test_run(self, data) -> str:
        """Create a new test run node and link it to the prompt version."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Convert metrics and input_params to JSON strings as Neo4j doesn't support nested objects as properties
                metrics_json = json.dumps(data.metrics.dict())
                input_params_json = json.dumps(data.input_params)
                
                result = session.run(
                    """
                    MATCH (pv:PromptVersion {id: $prompt_version_id})
                    CREATE (tr:TestRun {
                        id: randomUUID(),
                        model_used: $model_used,
                        output: $output,
                        metrics_json: $metrics_json,
                        input_params_json: $input_params_json,
                        created_at: datetime()
                    })
                    CREATE (tr)-[:TESTED_WITH]->(pv)
                    RETURN tr.id as id
                    """,
                    prompt_version_id=data.prompt_version_id,
                    model_used=data.model_used,
                    output=data.output,
                    metrics_json=metrics_json,
                    input_params_json=input_params_json
                )
                record = result.single()
                if record:
                    return record["id"]
                return None
        except Exception as e:
            logger.error(f"Error creating test run: {str(e)}")
            raise

    def get_test_runs(self, prompt_version_id: str) -> List[Dict]:
        """Get all test runs for a specific prompt version."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (tr:TestRun)-[:TESTED_WITH]->(pv:PromptVersion {id: $prompt_version_id})
                    RETURN tr {
                        .*
                    } as test_run
                    ORDER BY tr.created_at DESC
                    """,
                    prompt_version_id=prompt_version_id
                )
                test_runs = [record["test_run"] for record in result]
                test_runs = [self._convert_neo4j_values(test_run) for test_run in test_runs]
                
                # Parse JSON strings back into dictionaries
                for test_run in test_runs:
                    if 'metrics_json' in test_run:
                        test_run['metrics'] = json.loads(test_run['metrics_json'])
                        del test_run['metrics_json']
                    if 'input_params_json' in test_run:
                        test_run['input_params'] = json.loads(test_run['input_params_json'])
                        del test_run['input_params_json']
                
                return test_runs
        except Exception as e:
            logger.error(f"Error getting test runs: {str(e)}")
            raise
            
    def get_prompt_lineage(self, prompt_version_id: str) -> List[Dict]:
        """Get the full lineage (ancestry) of a prompt version."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH path = (pv:PromptVersion {id: $prompt_version_id})-[:DERIVED_FROM*]->(ancestor:PromptVersion)
                    RETURN ancestor {
                        .*,
                        depth: length(path)
                    } as ancestor
                    ORDER BY length(path)
                    """,
                    prompt_version_id=prompt_version_id
                )
                lineage = [record["ancestor"] for record in result]
                return [self._convert_neo4j_values(ancestor) for ancestor in lineage]
        except Exception as e:
            logger.error(f"Error getting prompt lineage: {str(e)}")
            raise
            
    def compare_test_runs(self, test_run_ids: List[str]) -> List[Dict]:
        """Compare multiple test runs side by side."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (tr:TestRun)
                    WHERE tr.id IN $test_run_ids
                    MATCH (tr)-[:TESTED_WITH]->(pv:PromptVersion)
                    MATCH (pv)-[:VERSION_OF]->(p:Prompt)
                    RETURN tr {
                        .*,
                        prompt_version: {
                            id: pv.id,
                            content: pv.content,
                            version: pv.version,
                            prompt: {
                                id: p.id,
                                name: p.name
                            }
                        }
                    } as test_run
                    """,
                    test_run_ids=test_run_ids
                )
                test_runs = [record["test_run"] for record in result]
                test_runs = [self._convert_neo4j_values(test_run) for test_run in test_runs]
                
                # Parse JSON strings back into dictionaries
                for test_run in test_runs:
                    if 'metrics_json' in test_run:
                        test_run['metrics'] = json.loads(test_run['metrics_json'])
                        del test_run['metrics_json']
                    if 'input_params_json' in test_run:
                        test_run['input_params'] = json.loads(test_run['input_params_json'])
                        del test_run['input_params_json']
                
                return test_runs
        except Exception as e:
            logger.error(f"Error comparing test runs: {str(e)}")
            raise
            
    def search_test_runs(self, query: str, model_filter: Optional[str] = None) -> List[Dict]:
        """Search for test runs containing specific text in their output."""
        try:
            # Rename the parameter to avoid conflict with the Cypher query argument
            query_params = {"search_text": f"*{query}*"}
            model_clause = ""
            if model_filter:
                query_params["model"] = model_filter
                model_clause = "AND tr.model_used = $model"

            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes("test_run_output", $search_text) 
                    YIELD node as tr, score
                    WHERE tr:TestRun {model_clause}
                    MATCH (tr)-[:TESTED_WITH]->(pv:PromptVersion)
                    MATCH (pv)-[:VERSION_OF]->(p:Prompt)
                    RETURN tr {{
                        .*,
                        score: score,
                        prompt_version: {{
                            id: pv.id,
                            content: pv.content,
                            version: pv.version,
                            prompt: {{
                                id: p.id,
                                name: p.name
                            }}
                        }}
                    }} as test_run
                    ORDER BY score DESC
                    LIMIT 20
                    """,
                    **query_params
                )
                test_runs = [record["test_run"] for record in result]
                test_runs = [self._convert_neo4j_values(test_run) for test_run in test_runs]
                
                # Parse JSON strings back into dictionaries
                for test_run in test_runs:
                    if 'metrics_json' in test_run:
                        test_run['metrics'] = json.loads(test_run['metrics_json'])
                        del test_run['metrics_json']
                    if 'input_params_json' in test_run:
                        test_run['input_params'] = json.loads(test_run['input_params_json'])
                        del test_run['input_params_json']
                
                return test_runs
        except Exception as e:
            logger.error(f"Error searching test runs: {str(e)}")
            raise
    
    def create_indexes(self):
        """Create necessary indexes and constraints for the database."""
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Create constraints
                session.run("CREATE CONSTRAINT prompt_group_id IF NOT EXISTS FOR (pg:PromptGroup) REQUIRE pg.id IS UNIQUE")
                session.run("CREATE CONSTRAINT prompt_id IF NOT EXISTS FOR (p:Prompt) REQUIRE p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT prompt_version_id IF NOT EXISTS FOR (pv:PromptVersion) REQUIRE pv.id IS UNIQUE")
                session.run("CREATE CONSTRAINT test_run_id IF NOT EXISTS FOR (tr:TestRun) REQUIRE tr.id IS UNIQUE")
                
                # Create fulltext index for search
                session.run("""
                CREATE FULLTEXT INDEX test_run_output IF NOT EXISTS
                FOR (tr:TestRun)
                ON EACH [tr.output]
                """)
                
                logger.info("Successfully created Neo4j indexes and constraints")
        except Exception as e:
            logger.error(f"Error creating Neo4j indexes: {str(e)}")
            raise