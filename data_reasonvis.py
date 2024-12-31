import openai
import json
import time
import pandas as pd
import os
import random
from typing import Dict, List, Union, Optional
import requests
from tqdm import tqdm
import re


class ReasoningDatasetGenerator:
    def __init__(self, api_key: str, base_url: str, db_path: str):
        """
        初始化推理数据集生成器
        """
        self.api_key = api_key
        self.base_url = base_url
        self.db_path = db_path
        
        # 添加标准化规则
        self.field_name_standards = {
            # 聚合函数标准化
            'COUNT(*)': ['count', 'total', 'number', 'frequency'],
            'AVG': ['average', 'mean', 'avg'],
            'MIN': ['minimum', 'min', 'lowest'],
            'MAX': ['maximum', 'max', 'highest'],
            'SUM': ['total', 'sum'],
            
            # 常见字段名标准化
            'name': ['track_name', 'ship_name', 'station_name'],
            'count': ['race_count', 'ship_count', 'flight_count'],
            'total_flights': ['number_of_flights', 'flight_count'],
            'race_count': ['number_of_races', 'total_races'],
            'ship_count': ['number_of_ships', 'total_ships']
        }
        
        # 反向映射表
        self.field_mapping = {}
        for standard, variants in self.field_name_standards.items():
            for variant in variants:
                self.field_mapping[variant.lower()] = standard
        
    def _load_db_tables(self, db_id: str) -> Dict[str, pd.DataFrame]:
        """
        加载特定数据库的所有相关表格
        """
        db_tables = {}
        db_folder = os.path.join(self.db_path, db_id)
        
        try:
            if not os.path.exists(db_folder):
                print(f"Database folder not found: {db_folder}")
                return {}
                
            for file_name in os.listdir(db_folder):
                if file_name.endswith('.csv'):
                    table_name = file_name[:-4]
                    file_path = os.path.join(db_folder, file_name)
                    df = pd.read_csv(file_path)
                    db_tables[table_name] = df
                    
            return db_tables
        except Exception as e:
            print(f"Error loading database {db_id}: {str(e)}")
            return {}

    def _get_irrelevant_tables(self, db_tables: Dict[str, pd.DataFrame], used_tables: List[str]) -> List[str]:
        """
        获取未使用的表格名称，确保不包含SQL中使用的表
        """
        all_tables = set(db_tables.keys())
        # 转换表名为小写进行比较
        used_tables = set(t.lower() for t in used_tables)
        unused_tables = []
        
        for table in all_tables:
            # 检查表名的所有可能形式（原始、小写、标题式）是否在已使用表中
            if (table.lower() not in used_tables and 
                table not in used_tables and 
                table.title() not in used_tables):
                unused_tables.append(table)
        
        return unused_tables

    def _get_relevant_tables(self, vis_data: Dict) -> List[str]:
        """
        获取SQL中使用的所有表名，包括JOIN的表
        """
        sql = vis_data['vis_query']['data_part']['sql_part'].upper()
        tables = []
        
        try:
            # 提取FROM和JOIN子句中的所有表
            if ' FROM ' in sql:
                # 分割SQL获取FROM之后的部分
                from_part = sql.split(' FROM ')[1]
                
                # 处理JOIN
                join_parts = from_part.replace(' LEFT JOIN ', ' JOIN ')\
                                    .replace(' RIGHT JOIN ', ' JOIN ')\
                                    .replace(' INNER JOIN ', ' JOIN ')\
                                    .replace(' OUTER JOIN ', ' JOIN ')\
                                    .split(' JOIN ')
                
                # 处理每个部分（FROM和每个JOIN）
                for part in join_parts:
                    # 移除ON及之后的条件
                    if ' ON ' in part:
                        part = part.split(' ON ')[0]
                    # 移除WHERE及之后的条件
                    if ' WHERE ' in part:
                        part = part.split(' WHERE ')[0]
                    # 移除GROUP BY及之后的内容
                    if ' GROUP BY ' in part:
                        part = part.split(' GROUP BY ')[0]
                    # 移除ORDER BY及之后的内容
                    if ' ORDER BY ' in part:
                        part = part.split(' ORDER BY ')[0]
                    
                    # 提取表名
                    table = part.strip().split(' ')[0]
                    # 处理带别名的情况
                    if ' AS ' in table:
                        table = table.split(' AS ')[0]
                    # 清理表名
                    table = ''.join(c for c in table if c.isalnum() or c == '_')
                    if table:
                        tables.append(table.strip())
        
        except Exception as e:
            print(f"Error parsing SQL: {str(e)}")
            return []
        
        # 移除重复并返回
        return list(set(t.lower() for t in tables if t))

    def _call_api(self, prompt: str) -> Dict:
        """
        调用API生成推理场景
        """
        url = "https://gptproxy.llmpaas.tencent.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    if "```json" in content:
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_str = content[start:end]
                            try:
                                parsed_json = json.loads(json_str)
                                # 如果结果包含在 reasoning_scenario 中，提取出来
                                if isinstance(parsed_json, dict) and 'reasoning_scenario' in parsed_json:
                                    return parsed_json['reasoning_scenario']
                                return parsed_json
                            except json.JSONDecodeError:
                                print(f"Error parsing JSON: {json_str}")
                                return None
                    else:
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            print(f"Error parsing content: {content}")
                            return None
                else:
                    print("No choices in response")
                    return None
            else:
                print(f"API request failed with status code {response.status_code}")
                return None

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

    # def _build_prompt(self, vis_data: Dict, db_tables: Dict[str, pd.DataFrame]) -> str:
    #     """
    #     构建提示
    #     """
    #     nl_query = vis_data['nl_queries'][0]
    #     chart_type = vis_data['chart'].upper()
    #     db_id = vis_data['db_id']
        
    #     table_desc = ""
    #     for table_name, df in db_tables.items():
    #         table_desc += f"\n{table_name} table:"
    #         table_desc += f"\nColumns: {', '.join(df.columns)}"
    #         table_desc += f"\nSample data:\n{df.head(3).to_string()}\n"
                
    #     relevant_tables = self._get_relevant_tables(vis_data)
    #     irrelevant_tables = self._get_irrelevant_tables(db_tables, relevant_tables)

    #     prompt = f"""Based on the following visualization scenario, please generate a new visualization scenario in the exact format specified:

    # Original Visualization:
    # Query: {nl_query}
    # Chart Type: {chart_type}
    # Database: {db_id}

    # Database Tables Information:
    # {table_desc}

    # Current Visualization Result:
    # X-axis data: {vis_data['vis_obj']['x_data']}
    # Y-axis data: {vis_data['vis_obj']['y_data']}

    # Please generate a new visualization scenario that follows this exact JSON format:
    # {{
    #     "vis_query": {{
    #         "vis_part": "Visualize {chart_type}",
    #         "data_part": {{
    #             "sql_part": "SQL query",
    #             "binning": ""
    #         }},
    #         "VQL": "Visualize {chart_type} <complete SQL query>"
    #     }},
    #     "chart": "{vis_data['chart']}",
    #     "hardness": "Difficulty level (Easy/Medium/Hard)",
    #     "db_id": "{db_id}",
    #     "vis_obj": {{
    #         "chart": "{vis_data['chart'].lower()}",
    #         "x_name": "x-axis field name",
    #         "y_name": "y-axis field name",
    #         "x_data": [["actual data values"]],
    #         "y_data": [["actual data values"]],
    #         "classify": [],
    #         "describe": "",
    #         "sort": null
    #     }},
    #     "nl_queries": [
    #         "Natural language query 1",
    #         "Natural language query 2",
    #         "Natural language query 3",
    #         "Natural language query 4"
    #     ],
    #     "irrelevant_tables": {json.dumps(irrelevant_tables)},
    #     "query_meta": [
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }}
    #     ]
    # }}

    # Requirements:
    # 1. Use actual data values, not placeholders
    # 2. Chart type in vis_part and VQL must be in UPPERCASE (e.g., PIE, BAR, LINE)
    # 3. Make sure SQL query is valid for the given database
    # 4. VQL must include the complete SQL query (same as sql_part)
    # 5. Include all required fields exactly as shown
    # 6. Generate exactly 4 nl_queries
    # 7. Keep the chart type consistent throughout the response
    # 8. Chart type in chart field should match original case
    # 9. Chart type in vis_obj should be lowercase
    # """
    #     return prompt

    # def _build_prompt(self, vis_data: Dict, db_tables: Dict[str, pd.DataFrame]) -> str:
    #     """
    #     构建提示
    #     """
    #     nl_query = vis_data['nl_queries'][0]
    #     chart_type = vis_data['chart'].upper()
    #     db_id = vis_data['db_id']
        
    #     table_desc = ""
    #     for table_name, df in db_tables.items():
    #         table_desc += f"\n{table_name} table:"
    #         table_desc += f"\nColumns: {', '.join(df.columns)}"
    #         table_desc += f"\nSample data:\n{df.head(3).to_string()}\n"
                
    #     relevant_tables = self._get_relevant_tables(vis_data)
    #     irrelevant_tables = self._get_irrelevant_tables(db_tables, relevant_tables)

    #     prompt = f"""Based on the following visualization scenario, please generate a new visualization scenario in the exact format specified:

    # Original Visualization:
    # Query: {nl_query}
    # Chart Type: {chart_type}
    # Database: {db_id}

    # Database Tables Information:
    # {table_desc}

    # Current Visualization Result:
    # X-axis data: {vis_data['vis_obj']['x_data']}
    # Y-axis data: {vis_data['vis_obj']['y_data']}

    # Please generate a new visualization scenario that follows this exact JSON format:
    # {{
    #     "vis_query": {{
    #         "vis_part": "Visualize {chart_type}",
    #         "data_part": {{
    #             "sql_part": "SQL query",
    #             "binning": ""
    #         }},
    #         "VQL": "Visualize {chart_type} <complete SQL query>"
    #     }},
    #     "chart": "{vis_data['chart']}",
    #     "hardness": "Difficulty level (Easy/Medium/Hard)",
    #     "db_id": "{db_id}",
    #     "vis_obj": {{
    #         "chart": "{vis_data['chart'].lower()}",
    #         "x_name": "x-axis field name (categorical/grouping field from SQL)",
    #         "y_name": "y-axis field name (typically the aggregated/measure field)",
    #         "x_data": [["categorical/grouping values"]],
    #         "y_data": [["measure values"]],
    #         "classify": ["additional grouping field for stacked charts, empty array otherwise"],
    #         "describe": "",
    #         "sort": "asc/desc if ORDER BY is used, null otherwise"
    #     }},
    #     "nl_queries": [
    #         "Natural language query 1",
    #         "Natural language query 2",
    #         "Natural language query 3",
    #         "Natural language query 4"
    #     ],
    #     "irrelevant_tables": {json.dumps(irrelevant_tables)},
    #     "query_meta": [
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }},
    #         {{
    #             "channel_specified": ["x", "y"]
    #         }}
    #     ]
    # }}

    # Requirements:
    # 1. Use actual data values, not placeholders
    # 2. Chart type in vis_part and VQL must be in UPPERCASE (e.g., PIE, BAR, LINE)
    # 3. Make sure SQL query is valid for the given database
    # 4. VQL must include the complete SQL query (same as sql_part)
    # 5. Include all required fields exactly as shown
    # 6. Generate exactly 4 nl_queries
    # 7. Keep the chart type consistent throughout the response
    # 8. Chart type in chart field should match original case
    # 9. Chart type in vis_obj should be lowercase

    # Additional Visualization Requirements:
    # 1. For x_name and y_name: Use the exact field names from your SQL query (after AS if alias is used)
    # 2. The x-axis should typically be the categorical/grouping field, while the y-axis should be the measure/aggregated field
    # 3. For stacked charts (e.g., STACKED BAR):
    # - The second GROUP BY field should be included in the classify array
    # - The y-axis should always be the aggregated measure
    # 4. For regular charts:
    # - The classify array should be empty
    # 5. Sort direction:
    # - Include "asc" or "desc" if ORDER BY is used in SQL
    # - Use null if no ORDER BY clause
    # 6. Field names should be consistent between SQL and vis_obj
    # 7. Make sure x_data contains the categorical values and y_data contains the corresponding measure values

    # Example for a BAR chart:
    # SQL: SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department ORDER BY employee_count DESC
    # x_name should be "department" (categorical field)
    # y_name should be "employee_count" (measure field)
    # sort should be "desc"

    # Example for a STACKED BAR chart:
    # SQL: SELECT department, role, COUNT(*) as employee_count FROM employees GROUP BY department, role
    # x_name should be "department" (first grouping field)
    # y_name should be "employee_count" (measure field)
    # classify should be ["role"] (second grouping field)
    # """
    #     return prompt

    # def _build_prompt(self, vis_data: Dict, db_tables: Dict[str, pd.DataFrame]) -> str:
    #     """
    #     构建提示 - 专注于生成具有不同推理级别的可视化场景
    #     """
    #     # 获取数据库信息
    #     db_id = vis_data['db_id']
    #     table_desc = ""
    #     for table_name, df in db_tables.items():
    #         table_desc += f"\n{table_name} table:"
    #         table_desc += f"\nColumns: {', '.join(df.columns)}"
    #         table_desc += f"\nSample data:\n{df.head(3).to_string()}\n"
                
    #     relevant_tables = self._get_relevant_tables(vis_data)
    #     irrelevant_tables = self._get_irrelevant_tables(db_tables, relevant_tables)

    #     # 定义不同推理级别的提示模板
    #     reasoning_levels = {
    #         "L1": "Basic visualization generation - direct mapping from data to visuals",
    #         "L2": "Multi-step analytical reasoning - requires intermediate analysis",
    #         "L3": "Multi-view synthesis - involves multiple coordinated visualizations",
    #         "L4": "Context-dependent analysis - builds upon previous analytical context"
    #     }
        
    #     # 随机选择一个推理级别,但保持分布
    #     # L1: 30%, L2: 30%, L3: 25%, L4: 15%
    #     level_weights = [0.3, 0.3, 0.25, 0.15]
    #     chosen_level = random.choices(list(reasoning_levels.keys()), weights=level_weights)[0]

    #     prompt = f"""Given the following database information, generate a visualization scenario that demonstrates {chosen_level} level reasoning:

    # Database: {db_id}
    # Tables Information:
    # {table_desc}

    # Requirements for the visualization scenario:

    # 1. Reasoning Level: {chosen_level}
    # - {reasoning_levels[chosen_level]}

    # 2. Visualization Requirements:
    # - Choose appropriate chart type based on data characteristics and analytical needs
    # - Ensure clear mapping between data fields and visual elements
    # - Consider data aggregation and transformation needs

    # 3. Natural Language Queries:
    # - Generate 4 related queries that demonstrate progressive analysis
    # - Each query should build upon insights from previous steps
    # - Maintain analytical context across the sequence

    # Please generate a response in the following JSON format:
    # {
    #     "vis_query": {
    #         "vis_part": "Visualize <CHART_TYPE>",
    #         "data_part": {
    #             "sql_part": "<SQL_QUERY>",
    #             "binning": ""
    #         },
    #         "VQL": "Visualize <CHART_TYPE> <SQL_QUERY>"
    #     },
    #     "chart": "<Chart_Type>",
    #     "hardness": "<Easy/Medium/Hard>",
    #     "db_id": "{db_id}",
    #     "vis_obj": {
    #         "chart": "<chart_type_lowercase>",
    #         "x_name": "<x_axis_field>",
    #         "y_name": "<y_axis_field>",
    #         "x_data": ["<actual_values>"],
    #         "y_data": ["<actual_values>"],
    #         "classify": ["<grouping_fields_if_needed>"],
    #         "describe": "<analysis_context>",
    #         "sort": "<sort_direction_if_applicable>"
    #     },
    #     "nl_queries": [
    #         "<Progressive_Query_1>",
    #         "<Progressive_Query_2>",
    #         "<Progressive_Query_3>",
    #         "<Progressive_Query_4>"
    #     ],
    #     "irrelevant_tables": {json.dumps(irrelevant_tables)},
    #     "query_meta": [
    #         {{"channel_specified": ["x", "y"]}},
    #         {{"channel_specified": ["x", "y"]}},
    #         {{"channel_specified": ["x", "y"]}},
    #         {{"channel_specified": ["x", "y"]}}
    #     ]
    # }

    # Guidelines for different reasoning levels:

    # L1 (Basic):
    # - Direct visualization of data fields
    # - Simple aggregations and groupings
    # - Clear one-to-one mapping of data to visuals

    # L2 (Multi-step):
    # - Include intermediate analysis steps
    # - Build upon basic visualizations
    # - Incorporate data comparisons or trends

    # L3 (Multi-view):
    # - Coordinate multiple visualizations
    # - Synthesize insights across views
    # - Show relationships between different aspects

    # L4 (Context-dependent):
    # - Reference previous analysis steps
    # - Adapt visualizations based on findings
    # - Support complex analytical reasoning

    # Requirements:
    # 1. Use actual data values, not placeholders. Generated values should be realistic and consistent
    # 2. Chart type in vis_part and VQL must be in UPPERCASE (e.g., PIE, BAR, LINE). Chart type in vis_obj should be lowercase
    # 3. Ensure SQL query is valid for the given database schema
    # 4. VQL must include the complete SQL query (same as sql_part)
    # 5. Include all required fields exactly as shown
    # 6. Generate exactly 4 nl_queries
    # 7. Keep the chart type consistent throughout the response
    # 8. Chart type in chart field should match original case and the analytical requirements
    # 9. Field names must match database columns
    # 10. Progressive queries should follow logical analysis flow

    # Additional Visualization Requirements:
    # 1. For x_name and y_name: Use the exact field names from your SQL query (after AS if alias is used)
    # 2. The x-axis should typically be the categorical/grouping field, while the y-axis should be the measure/aggregated field
    # 3. For stacked charts (e.g., STACKED BAR):
    # - The second GROUP BY field should be included in the classify array
    # - The y-axis should always be the aggregated measure
    # 4. For regular charts:
    # - The classify array should be empty
    # 5. Sort direction:
    # - Include "asc" or "desc" if ORDER BY is used in SQL
    # - Use null if no ORDER BY clause
    # 6. Field names should be consistent between SQL and vis_obj
    # 7. Make sure x_data contains the categorical values and y_data contains the corresponding measure values

    
    # """
    #     return prompt

    def _build_prompt(self, vis_data: Dict, db_tables: Dict[str, pd.DataFrame]) -> str:
        """
        构建提示 - 专注于生成具有不同推理级别的可视化场景
        """
        # 获取数据库信息
        db_id = vis_data['db_id']
        
        # 构建表描述
        table_desc = self._build_table_description(db_tables)
                
        # 获取相关和不相关表
        relevant_tables = self._get_relevant_tables(vis_data)
        irrelevant_tables = self._get_irrelevant_tables(db_tables, relevant_tables)
        irrelevant_tables_json = json.dumps(irrelevant_tables)

        # 定义推理级别
        reasoning_levels = {
            "L1": "Basic visualization generation - direct mapping from data to visuals",
            "L2": "Multi-step analytical reasoning - requires intermediate analysis",
            "L3": "Multi-view synthesis - involves multiple coordinated visualizations",
            "L4": "Context-dependent analysis - builds upon previous analytical context"
        }
        
        # 选择推理级别
        level_weights = [0.3, 0.3, 0.25, 0.15]
        chosen_level = random.choices(list(reasoning_levels.keys()), weights=level_weights)[0]
        chosen_level_desc = reasoning_levels[chosen_level]

        # 构建基础提示
        base_prompt = (
            f"Given the following database information, generate a visualization scenario "
            f"that demonstrates {chosen_level} level reasoning:\n\n"
            f"Database: {db_id}\n"
            f"Tables Information:\n{table_desc}\n\n"
            f"Requirements for the visualization scenario:\n\n"
            f"1. Reasoning Level: {chosen_level}\n"
            f"- {chosen_level_desc}\n\n"
        )

        # 构建可视化要求
        vis_requirements = (
            "2. Visualization Requirements:\n"
            "- Choose appropriate chart type based on data characteristics and analytical needs\n"
            "- Ensure clear mapping between data fields and visual elements\n"
            "- Consider data aggregation and transformation needs\n\n"
            "3. Natural Language Queries:\n"
            "- Generate 4 related queries that demonstrate progressive analysis\n"
            "- Each query should build upon insights from previous steps\n"
            "- Maintain analytical context across the sequence\n\n"
        )

        # 构建JSON格式模板
        json_template = (
            "Please generate a response in the following JSON format:\n"
            "{\n"
            '    "vis_query": {\n'
            '        "vis_part": "Visualize <CHART_TYPE>",\n'
            '        "data_part": {\n'
            '            "sql_part": "<SQL_QUERY>",\n'
            '            "binning": ""\n'
            "        },\n"
            '        "VQL": "Visualize <CHART_TYPE> <SQL_QUERY>"\n'
            "    },\n"
            f'    "reasoning_level": "{chosen_level}",\n'  # 添加推理级别
            '    "chart": "<Chart_Type>",\n'
            '    "hardness": "<Easy/Medium/Hard>",\n'
            f'    "db_id": "{db_id}",\n'
            '    "vis_obj": {\n'
            '        "chart": "<chart_type_lowercase>",\n'
            '        "x_name": "<x_axis_field>",\n'
            '        "y_name": "<y_axis_field>",\n'
            '        "x_data": ["<actual_values>"],\n'
            '        "y_data": ["<actual_values>"],\n'
            '        "classify": ["<grouping_fields_if_needed>"],\n'
            '        "describe": "<analysis_context>",\n'
            '        "sort": "<sort_direction_if_applicable>"\n'
            "    },\n"
            '    "nl_queries": [\n'
            '        "<Progressive_Query_1>",\n'
            '        "<Progressive_Query_2>",\n'
            '        "<Progressive_Query_3>",\n'
            '        "<Progressive_Query_4>"\n'
            "    ],\n"
            f'    "irrelevant_tables": {irrelevant_tables_json},\n'
            '    "query_meta": [\n'
            '        {"channel_specified": ["x", "y"]},\n'
            '        {"channel_specified": ["x", "y"]},\n'
            '        {"channel_specified": ["x", "y"]},\n'
            '        {"channel_specified": ["x", "y"]}\n'
            "    ]\n"
            "}\n\n"
        )

        # 构建级别指南
        level_guidelines = (
            "Guidelines for different reasoning levels:\n\n"
            "L1 (Basic):\n"
            "- Direct visualization of data fields\n"
            "- Simple aggregations and groupings\n"
            "- Clear one-to-one mapping of data to visuals\n\n"
            "L2 (Multi-step):\n"
            "- Include intermediate analysis steps\n"
            "- Build upon basic visualizations\n"
            "- Incorporate data comparisons or trends\n\n"
            "L3 (Multi-view):\n"
            "- Coordinate multiple visualizations\n"
            "- Synthesize insights across views\n"
            "- Show relationships between different aspects\n\n"
            "L4 (Context-dependent):\n"
            "- Reference previous analysis steps\n"
            "- Adapt visualizations based on findings\n"
            "- Support complex analytical reasoning\n\n"
        )

        # 构建重要注意事项
        important_notes = (
            "Important Notes:\n"
            "1. Use actual data values, not placeholders. Generated values should be realistic and consistent\n"
            "2. Chart type in vis_part and VQL must be in UPPERCASE (e.g., PIE, BAR, LINE). Chart type in vis_obj should be lowercase\n"
            "3. Ensure SQL query is valid for the given database schema\n"
            "4. VQL must include the complete SQL query (same as sql_part)\n"
            "5. Include all required fields exactly as shown\n"
            "6. Generate exactly 4 nl_queries\n"
            "7. Keep the chart type consistent throughout the response\n"
            "8. Chart type in chart field should match original case and the analytical requirements\n"
            "9. Field names must match database columns\n"
            "10. Progressive queries should follow logical analysis flow\n"

            "Additional Visualization Requirements:\n"
            "1. For x_name and y_name: Use the exact field names from your SQL query (after AS if alias is used)\n"
            "2. The x-axis should typically be the categorical/grouping field, while the y-axis should be the measure/aggregated field\n"
            "3. For stacked charts (e.g., STACKED BAR):\n"
            "- The second GROUP BY field should be included in the classify array\n"
            "- The y-axis should always be the aggregated measure\n"
            "4. For regular charts:\n"
            "- The classify array should be empty\n"
            "5. Sort direction:\n"
            "- Include asc or desc if ORDER BY is used in SQL\n"
            "- Use null if no ORDER BY clause\n"
            "6. Field names should be consistent between SQL and vis_obj\n"
            "7. Make sure x_data contains the categorical values and y_data contains the corresponding measure values\n"

            
        )


        # 组合所有部分
        prompt = (
            base_prompt + 
            vis_requirements + 
            json_template + 
            level_guidelines + 
            important_notes
        )

        return prompt

    def _build_table_description(self, db_tables: Dict[str, pd.DataFrame]) -> str:
        """构建数据库表描述"""
        table_desc = ""
        for table_name, df in db_tables.items():
            table_desc += f"\n{table_name} table:"
            table_desc += f"\nColumns: {', '.join(df.columns)}"
            table_desc += f"\nSample data:\n{df.head(3).to_string()}\n"
        return table_desc

    def clean_vis_id(self, vis_id: str) -> str:
        """
        清理可视化ID，只保留基础ID
        """
        return vis_id.split('@')[0]

    def _validate_chart_type(self, response: Dict) -> str:
        """验证并调整图表类型"""
        sql_part = response['vis_query']['data_part']['sql_part']
        chart_type = response['chart']
        
        # 检查是否有 GROUP BY 多个字段
        group_by_match = re.search(r'GROUP BY\s+([^ORDER]+)', sql_part, re.IGNORECASE)
        if group_by_match:
            group_by_count = len([f.strip() for f in group_by_match.group(1).split(',')])
            if group_by_count > 1:
                if chart_type.lower() == 'line':
                    return 'Grouping Line'
                elif chart_type.lower() == 'scatter':
                    return 'Grouping Scatter'
        
        # 检查是否适合使用折线图
        if chart_type.lower() == 'line':
            # 检查x轴是否为时间/年份类型
            has_time = any(word in sql_part.lower() for word in ['date', 'year', 'month', 'time'])
            # 如果不是时间类型且有ORDER BY，建议使用柱状图
            if not has_time and 'ORDER BY' in sql_part.upper():
                return 'Bar'
        
        return chart_type

    def _process_grouping(self, response: Dict) -> None:
        """处理分组信息"""
        sql_part = response['vis_query']['data_part']['sql_part']
        
        # 提取 GROUP BY 字段
        group_by_match = re.search(r'GROUP BY\s+([^ORDER]+)', sql_part, re.IGNORECASE)
        if group_by_match:
            group_fields = [f.strip() for f in group_by_match.group(1).split(',')]
            if len(group_fields) > 1:
                # 第一个字段通常用于 x 轴，其他字段用于分组
                classify_fields = [self._standardize_field_name(f, f, sql_part) for f in group_fields[1:]]
                response['vis_obj']['classify'] = classify_fields
                
                # 如果是Grouping类型的图表，确保有正确的分组数据
                if 'Grouping' in response['chart']:
                    # 需要根据分组字段重新组织数据
                    # 这里需要实现具体的数据重组逻辑
                    pass

    def _standardize_chart_type(self, chart_type: str) -> Dict[str, str]:
        """标准化图表类型"""
        chart_type = chart_type.lower()
        if 'grouping' in chart_type:
            base_type = chart_type.split()[-1]
            return {
                'upper': f'GROUPING {base_type.upper()}',
                'normal': f'Grouping {base_type.capitalize()}',
                'lower': base_type.lower()
            }
        return {
            'upper': chart_type.upper(),
            'normal': chart_type.capitalize(),
            'lower': chart_type.lower()
        }

    def _standardize_data_format(self, data: List) -> List[List]:
        """
        标准化数据格式，确保是单个嵌套数组
        """
        if not data:
            return [[]]
            
        # 如果是多层嵌套数组，将其展平为单个嵌套数组
        if isinstance(data[0], list):
            flattened = []
            for sublist in data:
                flattened.extend(sublist)
            return [flattened]
        
        # 如果是单层数组，包装为嵌套数组
        return [data]

    def _standardize_field_name(self, sql_field: str, vis_field: str, sql_part: str) -> str:
        """
        标准化字段名称，确保与SQL完全一致（包括大小写）
        """
        # 检查标准化映射
        field_lower = vis_field.lower().replace(' ', '_')
        if field_lower in self.field_mapping:
            mapped_field = self.field_mapping[field_lower]
            # 在SQL中查找映射后的字段
            if mapped_field.lower() in sql_part.lower():
                return mapped_field
        
        # 原有的处理逻辑
        if '.' in sql_field:
            sql_field = sql_field.split('.')[-1].strip()
        
        if ' AS ' in sql_field:
            return sql_field.split(' AS ')[1].strip()
        elif ' as ' in sql_field:
            return sql_field.split(' as ')[1].strip()
                
        if 'COUNT(' in sql_field.upper():
            match = re.search(r'COUNT\([^)]+\)\s+(?:AS|as)\s+(\w+)', sql_part, re.IGNORECASE)
            if match:
                return match.group(1)
            return sql_field
                    
        if '(' in sql_field:
            match = re.search(r'\w+\([^)]+\)\s+(?:AS|as)\s+(\w+)', sql_part, re.IGNORECASE)
            if match:
                return match.group(1)
            return sql_field
        
        return sql_field

    def _extract_sql_fields(self, sql_part: str) -> Dict[str, str]:
        """
        从SQL中提取字段和其标准化名称，保持大小写一致
        """
        fields = {}
        try:
            # 提取SELECT部分
            select_clause = sql_part[sql_part.upper().find('SELECT') + 6:sql_part.upper().find('FROM')].strip()
            
            # 分割字段
            raw_fields = []
            current_field = []
            in_parentheses = 0
            
            # 正确处理带括号的字段
            for char in select_clause:
                if char == '(':
                    in_parentheses += 1
                elif char == ')':
                    in_parentheses -= 1
                elif char == ',' and in_parentheses == 0:
                    raw_fields.append(''.join(current_field).strip())
                    current_field = []
                    continue
                current_field.append(char)
            raw_fields.append(''.join(current_field).strip())
            
            # 处理每个字段
            for field in raw_fields:
                # 获取标准化名称（保持大小写）
                std_name = self._standardize_field_name(field, '', sql_part)
                
                # 存储原始字段和标准化名称的映射
                fields[field] = std_name
                
                # 存储不区分大小写的变体用于匹配
                fields[std_name.lower()] = std_name
                fields[std_name.upper()] = std_name
                fields[std_name.replace('_', ' ').lower()] = std_name
                fields[std_name.replace('_', ' ').title()] = std_name
                
        except Exception as e:
            print(f"Error extracting SQL fields: {str(e)}")
            
        return fields

    def process_single_visualization(self, vis_data: Dict) -> Dict:
        """
        处理单个可视化场景
        """
        try:
            print("\nProcessing visualization:")
            
            db_id = vis_data.get('db_id')
            if not db_id:
                print("No database ID found in visualization data")
                return None
                
            print(f"Database ID: {db_id}")
            
            db_tables = self._load_db_tables(db_id)
            if not db_tables:
                print(f"No tables found for database: {db_id}")
                return None
            
            prompt = self._build_prompt(vis_data, db_tables)
            response = self._call_api(prompt)
            
            if response is None:
                print("Failed to get API response")
                return None
                
            # 标准化图表类型，并根据数据特征调整
            chart_types = self._standardize_chart_type(response['chart'])
            adjusted_chart = self._validate_chart_type(response)
            if adjusted_chart != response['chart']:
                chart_types = self._standardize_chart_type(adjusted_chart)
            
            response['chart'] = chart_types['normal']
            response['vis_query']['vis_part'] = f"Visualize {chart_types['upper']}"
            response['vis_obj']['chart'] = chart_types['lower']
            
            # 确保VQL格式一致
            if 'vis_query' in response:
                sql_part = response['vis_query']['data_part']['sql_part']
                binning = response['vis_query']['data_part'].get('binning', '')
                
                # 如果存在binning，将其插入到GROUP BY前
                if binning:
                    group_by_idx = sql_part.upper().find('GROUP BY')
                    if group_by_idx != -1:
                        sql_part = f"{sql_part[:group_by_idx]} BINNING {binning} {sql_part[group_by_idx:]}"
                        response['vis_query']['data_part']['sql_part'] = sql_part
                
                response['vis_query']['VQL'] = f"Visualize {chart_types['upper']} {sql_part}"
            
            # 标准化字段名称，确保与SQL一致
            if 'vis_query' in response and 'vis_obj' in response:
                sql_part = response['vis_query']['data_part']['sql_part']
                
                # 提取 SQL 中的字段名，保持原始大小写
                select_clause = re.findall(r'SELECT\s+(.+?)\s+FROM', sql_part, re.IGNORECASE)[0]
                fields = [f.strip() for f in select_clause.split(',')]
                
                # 处理 COUNT(*) 和其他聚合函数
                agg_pattern = re.compile(r'(COUNT|SUM|AVG|MAX|MIN)\s*\((.*?)\)(?:\s+(?:AS|as)\s+(\w+))?', re.IGNORECASE)
                for i, field in enumerate(fields):
                    match = agg_pattern.search(field)
                    if match:
                        func, expr, alias = match.groups()
                        if not alias:
                            if expr == '*':
                                alias = 'count'
                            else:
                                alias = f"{func.lower()}_{expr.strip().lower()}"
                        fields[i] = f"{func.upper()}({expr}) AS {alias}"
                
                sql_part = re.sub(r'SELECT\s+.+?\s+FROM', f'SELECT {", ".join(fields)} FROM', sql_part)
                response['vis_query']['data_part']['sql_part'] = sql_part
                
                # 更新字段名
                x_field = fields[0]
                y_field = fields[1] if len(fields) > 1 else 'COUNT(*) AS count'
                
                response['vis_obj']['x_name'] = self._standardize_field_name(
                    x_field, 
                    response['vis_obj']['x_name'],
                    sql_part
                )
                response['vis_obj']['y_name'] = self._standardize_field_name(
                    y_field,
                    response['vis_obj']['y_name'],
                    sql_part
                )
                
                # 处理分组信息
                self._process_grouping(response)
                
                # 标准化数据格式 - 移除一层嵌套
                if isinstance(response['vis_obj']['x_data'], list) and len(response['vis_obj']['x_data']) > 0:
                    if isinstance(response['vis_obj']['x_data'][0], list):
                        response['vis_obj']['x_data'] = response['vis_obj']['x_data'][0]
                        
                if isinstance(response['vis_obj']['y_data'], list) and len(response['vis_obj']['y_data']) > 0:
                    if isinstance(response['vis_obj']['y_data'][0], list):
                        response['vis_obj']['y_data'] = response['vis_obj']['y_data'][0]
                
                # 确保必要的字段存在
                if 'classify' not in response['vis_obj']:
                    response['vis_obj']['classify'] = []
                if 'describe' not in response['vis_obj']:
                    response['vis_obj']['describe'] = ""
                if 'sort' not in response['vis_obj']:
                    response['vis_obj']['sort'] = None
                    
                # 从ORDER BY子句推断排序方向
                order_match = re.search(r'ORDER BY\s+.+?\s+(DESC|ASC)', sql_part, re.IGNORECASE)
                if order_match:
                    response['vis_obj']['sort'] = order_match.group(1).lower()
            
            # 基本字段验证
            required_fields = ['vis_query', 'chart', 'hardness', 'db_id', 'vis_obj', 
                            'nl_queries', 'irrelevant_tables', 'query_meta']
            if not all(field in response for field in required_fields):
                print("Response missing required fields")
                return None
            
            # VQL格式验证
            if 'vis_query' in response:
                vql = response['vis_query'].get('VQL', '')
                sql_part = response['vis_query']['data_part'].get('sql_part', '')
                if 'SQL query' in vql or not sql_part in vql:
                    print("Invalid VQL format")
                    return None
                
            # 确保ID是字符串形式
            if isinstance(vis_data.get('vis_id'), int):
                vis_data['vis_id'] = str(vis_data['vis_id'])
                
            # 验证nl_queries和query_meta数量
            if (len(response.get('nl_queries', [])) != 4 or 
                len(response.get('query_meta', [])) != 4):
                print("Number of nl_queries or query_meta entries is not 4")
                return None
                
            # 确保每个query_meta都有正确的channel_specified
            for meta in response['query_meta']:
                if not isinstance(meta, dict) or 'channel_specified' not in meta:
                    print("Invalid query_meta format")
                    return None
                if not isinstance(meta['channel_specified'], list) or \
                not all(channel in ['x', 'y'] for channel in meta['channel_specified']):
                    print("Invalid channel_specified in query_meta")
                    return None
            
            return response
                
        except Exception as e:
            print(f"Error in process_single_visualization: {str(e)}")
            return None
        
    def generate_dataset(self, viseval_data: Dict, output_path: str):
        """
        生成完整的推理数据集
        """
        results = {}
        
        # 清理和去重ID
        clean_data = {}
        for vis_id, data in viseval_data.items():
            clean_id = self.clean_vis_id(vis_id)
            if clean_id not in clean_data:
                clean_data[clean_id] = data
        
        for vis_id, vis_data in tqdm(clean_data.items()):
            result = self.process_single_visualization(vis_data)
            if result:
                results[vis_id] = result
            time.sleep(1)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


    # def generate_sample_dataset(self,
    #                         viseval_data: Dict,
    #                         sample_size: int = 3,
    #                         output_path: str = 'sample_reasoning_dataset.json') -> Dict:
    #     """
    #     生成样本数据集
    #     """
    #     results = {}
    #     success_count = 0
        
    #     # 清理和去重ID
    #     clean_data = {}
    #     for vis_id, data in viseval_data.items():
    #         clean_id = self.clean_vis_id(vis_id)
    #         if clean_id not in clean_data:
    #             clean_data[clean_id] = data
        
    #     vis_ids = list(clean_data.keys())
    #     if not vis_ids:
    #         print("Error: No visualization IDs found in data")
    #         return results
            
    #     sample_ids = random.sample(vis_ids, min(sample_size, len(vis_ids)))
        
    #     with tqdm(total=len(sample_ids)) as pbar:
    #         for vis_id in sample_ids:
    #             print(f"\nProcessing visualization {vis_id}")
                
    #             try:
    #                 vis_data = clean_data[vis_id]
    #                 result = self.process_single_visualization(vis_data)
                    
    #                 if result is not None:
    #                     results[vis_id] = result
    #                     success_count += 1
    #                 else:
    #                     print(f"Failed to generate reasoning scenario for visualization {vis_id}")
                        
    #             except Exception as e:
    #                 print(f"Error processing visualization {vis_id}: {str(e)}")
    #                 continue
                    
    #             finally:
    #                 pbar.update(1)
        
    #     if results:
    #         with open(output_path, 'w', encoding='utf-8') as f:
    #             json.dump(results, f, indent=2, ensure_ascii=False)
                
    #         print(f"\nProcessed {success_count} visualizations successfully")
    #         print(f"Results saved to {output_path}")
            
    #         print("\nGenerated Sample Results:")
    #         print(json.dumps(results, indent=2))
    #     else:
    #         print("\nNo successful results generated")
            
    #     return results

    def generate_sample_dataset(self, viseval_data: Dict, sample_size: Optional[int] = None, output_path: str = 'reasonvis_dataset.json') -> Dict:
        """
        生成推理数据集
        Args:
            viseval_data: 原始数据
            sample_size: 采样大小,None表示处理所有数据
            output_path: 输出文件路径
        """
        results = {}
        total_items = len(viseval_data)
        
        # 如果sample_size为None,处理所有数据
        if sample_size is None:
            items_to_process = viseval_data
        else:
            # 随机采样
            sample_keys = random.sample(list(viseval_data.keys()), sample_size)
            items_to_process = {k: viseval_data[k] for k in sample_keys}
        
        print(f"Processing {len(items_to_process)} items...")
        
        for idx, (key, item) in enumerate(items_to_process.items(), 1):
            try:
                print(f"Processing item {idx}/{len(items_to_process)} (Key: {key})")
                
                # 加载相关数据库 - 修改这里的方法名
                db_id = item['db_id']
                db_tables = self._load_db_tables(db_id)  # 使用正确的方法名
                
                # 构建提示
                prompt = self._build_prompt(item, db_tables)
                
                # 调用API
                response = self._call_api(prompt)
                if response:
                    results[key] = response
                    
                # 定期保存中间结果
                if idx % 10 == 0:
                    self._save_results(results, output_path)
                    print(f"Intermediate results saved. Processed {idx}/{len(items_to_process)}")
                    
            except Exception as e:
                print(f"Error processing item {key}: {str(e)}")
                continue
        
        # 保存最终结果
        self._save_results(results, output_path)
        
        return results


    def _save_results(self, results: Dict, output_path: str):
        """
        保存结果到文件
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {str(e)}")

# def main():
#     api_key = "8beCPow2KcmVGufSecmUZrTQhVN2OnPb"
#     base_url = "https://gptproxy.llmpaas.tencent.com"
#     db_path = "databases"
    
#     generator = ReasoningDatasetGenerator(
#         api_key=api_key,
#         base_url=base_url,
#         db_path=db_path
#     )
    
#     test_prompt = """Generate a visualization scenario in JSON format with:
#     {
#         "vis_query": {
#             "vis_part": "Visualize CHART_TYPE",
#             "data_part": {
#                 "sql_part": "SQL query",
#                 "binning": ""
#             }
#         }
#     }"""
    
#     print("Testing API connection...")
#     test_response = generator._call_api(test_prompt)
#     if test_response is None:
#         print("Failed to connect to API. Please check your API key and connection.")
#         return
        
#     print("API connection successful!")
    
#     try:
#         with open('visEval.json', 'r', encoding='utf-8') as f:
#             viseval_data = json.load(f)
            
#         sample_results = generator.generate_sample_dataset(
#             viseval_data=viseval_data,
#             sample_size=10,
#             output_path='sample_reasoning_dataset.json'
#         )
        
#     except Exception as e:
#         print(f"Error in main execution: {str(e)}")
#         raise

def main():
    api_key = "8beCPow2KcmVGufSecmUZrTQhVN2OnPb"
    base_url = "https://gptproxy.llmpaas.tencent.com"
    db_path = "databases"
    
    generator = ReasoningDatasetGenerator(
        api_key=api_key,
        base_url=base_url,
        db_path=db_path
    )
    
    test_prompt = """Generate a visualization scenario in JSON format with:
    {
        "vis_query": {
            "vis_part": "Visualize CHART_TYPE",
            "data_part": {
                "sql_part": "SQL query",
                "binning": ""
            }
        }
    }"""
    
    print("Testing API connection...")
    test_response = generator._call_api(test_prompt)
    if test_response is None:
        print("Failed to connect to API. Please check your API key and connection.")
        return
        
    print("API connection successful!")
    
    try:
        # 加载原始数据
        with open('visEval.json', 'r', encoding='utf-8') as f:
            viseval_data = json.load(f)
        
        # 生成完整数据集
        print("Generating complete ReasonVis dataset...")
        complete_results = generator.generate_sample_dataset(
            viseval_data=viseval_data,
            sample_size=None,  # 设为None处理所有数据
            output_path='reasonvis_dataset.json'  # 修改输出文件名
        )
        
        print(f"Dataset generation completed. Results saved to reasonvis_dataset.json")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

