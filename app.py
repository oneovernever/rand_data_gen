"""
Gradio Random Data Generator with Llama3.2 (1B) via Ollama
Run:  python app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
from faker import Faker
import random, io, os, json, tempfile
import re
from datetime import datetime, timedelta

# -------- Agentic & validation imports --------------------------------------
from typing import List, Optional, Union, Any, Literal, Dict
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
# -----------------------------------------------------------------------------

# -------- Schema Analysis & Data Generation Agents -----------------------------
class SchemaRules(BaseModel):
    field_dependencies: Dict[str, List[str]]
    validation_rules: Dict[str, List[str]]
    generation_rules: Dict[str, List[str]]
    custom_logic: Dict[str, str]

class SchemaAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=SchemaRules)
        
        self.prompt = PromptTemplate.from_template(
            """You are a data schema analysis expert. Analyze the following columns and their descriptions to identify:
1. Dependencies between fields
2. Validation rules
3. Generation rules
4. Custom logic needed

Columns:
{columns}

Dataset Description: {description}

Return a JSON object with exactly these keys and formats:
{{
    "field_dependencies": {{
        "field_name": ["dependent_field1", "dependent_field2"]
    }},
    "validation_rules": {{
        "field_name": ["rule1", "rule2"]
    }},
    "generation_rules": {{
        "field_name": ["rule1", "rule2"]
    }},
    "custom_logic": {{
        "field_name": "logic description"
    }}
}}

Example response:
{{
    "field_dependencies": {{
        "age": ["dob"],
        "email": ["name"]
    }},
    "validation_rules": {{
        "dob": ["must be in past", "must be valid date"],
        "email": ["must be valid email format"]
    }},
    "generation_rules": {{
        "name": ["use appropriate gender names"],
        "email": ["derive from name"]
    }},
    "custom_logic": {{
        "age": "calculate from dob",
        "email": "lowercase(name) + '@' + random_domain()"
    }}
}}

Focus on logical relationships and data coherence.
{format_instructions}
"""
        )

    def analyze(self, description: str, columns: pd.DataFrame) -> SchemaRules:
        if not self.llm:
            # Return default rules if no LLM available
            return SchemaRules(
                field_dependencies={},
                validation_rules={},
                generation_rules={},
                custom_logic={}
            )
            
        # Format columns for prompt
        cols_str = columns.to_string() if isinstance(columns, pd.DataFrame) else str(columns)
        
        # Get rules from LLM
        prompt = self.prompt.format(
            columns=cols_str,
            description=description,
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            result = self.llm.invoke(prompt)
            rules = self.parser.parse(result)
            return rules
        except Exception as e:
            print(f"⚠️ Error analyzing schema: {e}")
            # Return default rules on error
            return SchemaRules(
                field_dependencies={},
                validation_rules={},
                generation_rules={},
                custom_logic={}
            )

class FieldGenerator:
    def __init__(self, llm, faker_instance=None):
        self.llm = llm
        self.faker = faker_instance or faker
        
        self.prompt = PromptTemplate.from_template(
            """Generate a valid value for the field "{field_name}" that satisfies these constraints:

Field Specification:
{field_spec}

Rules to Follow:
{rules}

Current Values of Other Fields:
{current_state}

Return ONLY the generated value, no explanation.
"""
        )
    
    def generate_value(self, 
                      field_name: str, 
                      field_spec: Dict, 
                      rules: Dict, 
                      current_state: Dict) -> Any:
        if not self.llm:
            return None
            
        prompt = self.prompt.format(
            field_name=field_name,
            field_spec=json.dumps(field_spec, indent=2),
            rules=json.dumps(rules, indent=2),
            current_state=json.dumps(current_state, indent=2)
        )
        
        try:
            result = self.llm.invoke(prompt).strip()
            return result
        except Exception as e:
            print(f"⚠️ Error generating value for {field_name}: {e}")
            return None
# -----------------------------------------------------------------------------

# ----- Ollama configuration --------------------------------------------------
OLLAMA_API   = "http://localhost:11434"   # default Ollama port
OLLAMA_MODEL = "llama3.2:1B"          # change to q4_k_m etc. if desired
# -----------------------------------------------------------------------------

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None  # allow app to run without LangChain installed

llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_API, temperature=0.1) if Ollama else None

faker = Faker()

# ------------------- Pydantic schema for column spec ------------------------
AllowedType = Literal["integer", "float", "categorical", "text", "date"]

class ColumnSpec(BaseModel):
    name: str
    type: AllowedType
    description: str
    min: Optional[Union[int, float, str]] = None
    max: Optional[Union[int, float, str]] = None
    categories: Optional[List[str]] = None
    example: Optional[Any] = None
# -----------------------------------------------------------------------------

# ---------------------- LangChain agentic chains ----------------------------
if llm:
    name_parser = CommaSeparatedListOutputParser()
    name_prompt = PromptTemplate.from_template(
        """You are a data architect.
Given the dataset description below, list the MOST relevant column names, comma‑separated, no extra words.

Dataset: {task}

{format_instructions}"""
    ).partial(format_instructions=name_parser.get_format_instructions())
    propose_columns_chain = name_prompt | llm | name_parser

    enrich_prompt = PromptTemplate.from_template(
        """You are a JSON‑only assistant specialized in data schemas.

For the column "{col}" in the dataset described as "{task}", output ONE valid JSON object with exactly these keys:

{{
  "name":        "{col}",
  "type":        "integer" / "float" / "categorical" / "text" / "date",
  "description": "<short human description>",
  "min":         <number OR "YYYY-MM-DD" OR null>,
  "max":         <number OR "YYYY-MM-DD" OR null>,
  "categories":  <array of strings OR null>,
  "example":     <example value>
}}

**Strict Type Rules:**
1. Use "date" for:
   - Any date/time field (dob, signup_date, etc.)
   - min/max must be "YYYY-MM-DD" format
   - categories must be null
   - example must be a valid date string

2. Use "categorical" for:
   - Fields with a fixed set of possible values
   - min/max must be null
   - categories must be a relevant array of strings
   - example must be one of the categories

3. Use "integer" for:
   - Whole numbers only (counts, quantities)
   - min/max must be numbers or null
   - categories must be null
   - example must be a whole number
   - Must be logically valid (e.g., can't have 10.5 bets)

4. Use "float" for:
   - Decimal numbers (money, measurements)
   - min/max must be numbers or null
   - categories must be null
   - example must be a decimal number

5. Use "text" for:
   - Names, descriptions, free text
   - min/max must be null
   - categories can be null or relevant options
   - example must be a string

**Logical Consistency Rules:**
1. Money fields (deposit, withdraw, win) should be "float" type
2. Count fields (bets, visits) must be "integer" type
3. Date fields must use "date" type with valid date constraints
4. Gender should be "categorical" with appropriate categories
5. Example values must be logically valid for the field's purpose
6. Min/max ranges must make sense for the data type
7. Categories must only be used for truly categorical data

Output only the JSON object — no markdown, no code fencing.
"""
    )

    import json
    from pydantic_core import ValidationError as PydanticValidationError

    def _parse_user_columns(text: str) -> list[str] | None:
        """
        If the user prompt looks like an explicit column list (comma/semicolon‑separated),
        return the list; otherwise return None.
        """
        if re.search(r"[;,]", text):
            parts = re.split(r"[;,]\s*", text.strip())
            cols  = [p for p in parts if p]
            return cols or None
        return None

    def get_column_specs(task: str) -> List[ColumnSpec]:
        explicit = _parse_user_columns(task)
        names = explicit if explicit else propose_columns_chain.invoke({"task": task})
        specs: List[ColumnSpec] = []
        for n in names:
            raw = llm.invoke(enrich_prompt.format(col=n, task=task)).strip()
            print("🔄 RAW:", raw)

            # --- minimal cleanup -------------------------------------------------
            cleaned = raw.replace("<column name>", n).rstrip().rstrip(",")
            if cleaned.count("{") > cleaned.count("}"):
                cleaned += "}"
            cleaned = cleaned.replace('"string"', '"text"')
            # ---------------------------------------------------------------------

            try:
                obj = json.loads(cleaned)
                # map common alias
                if obj.get("type") == "string":
                    obj["type"] = "text"
                spec = ColumnSpec.model_validate(obj)
                specs.append(spec)
            except (json.JSONDecodeError, PydanticValidationError) as e:
                print("⚠️  Could not parse column", n, "→", e)
        return specs
else:
    def get_column_specs(task: str):
        return []
# -----------------------------------------------------------------------------

SPEC_COLUMNS = ["name", "type", "description", "min", "max", "categories", "example"]

def suggest_columns(task: str) -> pd.DataFrame:
    specs = [s.model_dump() for s in get_column_specs(task)] if task else []
    df = pd.DataFrame(specs)
    for col in SPEC_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SPEC_COLUMNS]

# ----------------------- Data generation helpers ----------------------------
def rand_date(start, end):
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate_data(spec_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    # Initialize agents
    analyzer = SchemaAnalyzer(llm)
    generator = FieldGenerator(llm, faker)
    
    # Get rules from schema analyzer
    rules = analyzer.analyze(
        description="Generate synthetic data following logical relationships",
        columns=spec_df
    )
    
    data: dict[str, list[Any]] = {}
    
    # First pass: generate data following dependencies
    processed_fields = set()
    
    def process_field(field_name: str):
        if field_name in processed_fields:
            return
            
        # Process dependencies first
        deps = rules.field_dependencies.get(field_name, [])
        for dep in deps:
            if dep not in processed_fields:
                process_field(dep)
        
        row = spec_df[spec_df['name'] == field_name].iloc[0]
        ctype = str(row.get("type", "text")).lower()
        
        # Map SQL-like types to our internal types
        type_mapping = {
            "varchar": "text",
            "dte": "date",
            "date": "date",
            "categorical": "categorical",
            "integer": "integer",
            "int": "integer",
            "float": "float",
            "text": "text"
        }
        ctype = type_mapping.get(ctype, "text")
        
        field_values = []
        for i in range(n_rows):
            # Get current state of other fields for this row
            current_state = {
                k: v[i] if isinstance(v, list) else v
                for k, v in data.items()
            }
            
            # Try to get value from field generator first
            value = generator.generate_value(
                field_name=field_name,
                field_spec={
                    "name": field_name,
                    "type": ctype,
                    "min": row.get("min"),
                    "max": row.get("max"),
                    "categories": row.get("categories"),
                    "example": row.get("example")
                },
                rules={
                    "validation": rules.validation_rules.get(field_name, []),
                    "generation": rules.generation_rules.get(field_name, []),
                    "custom": rules.custom_logic.get(field_name)
                },
                current_state=current_state
            )
            
            # Fallback to basic generation if needed
            if value is None:
                if ctype == "text":
                    if "name" in field_name.lower():
                        gender = current_state.get("gender", "").lower()
                        if gender == "female":
                            value = faker.name_female()
                        elif gender == "male":
                            value = faker.name_male()
                        else:
                            value = faker.name()
                    else:
                        ex = row.get("example")
                        value = ex if ex and ex != "Example value" else faker.word()
                
                elif ctype == "categorical":
                    cats = row.get("categories")
                    if isinstance(cats, str):
                        cats = [c.strip() for c in cats.split(",") if c.strip()]
                    if not cats:
                        example = row.get("example")
                        cats = [str(example)] if example and example != "Example value" else ["A", "B", "C"]
                    value = np.random.choice(cats)
                
                elif ctype == "date":
                    try:
                        start = row.get("min") or "1950-01-01"
                        end = row.get("max") or "2025-01-01"
                        start = "1950-01-01" if "YYYY" in str(start) else start
                        end = "2025-01-01" if "YYYY" in str(end) else end
                        
                        date = rand_date(start, end).date()
                        
                        # Apply basic validation rules
                        if field_name.lower() == "dob" and "signup_date" in data:
                            signup = datetime.strptime(data["signup_date"][i], "%Y-%m-%d").date()
                            while date >= signup:
                                date = rand_date(start, end).date()
                        elif field_name.lower() == "signup_date" and "dob" in data:
                            dob = datetime.strptime(data["dob"][i], "%Y-%m-%d").date()
                            while date <= dob:
                                date = rand_date(start, end).date()
                        
                        value = str(date)
                    except (ValueError, TypeError):
                        value = str(rand_date("1950-01-01", "2025-01-01").date())
                
                elif ctype == "integer":
                    try:
                        low = int(row.get("min", 0) or 0)
                        high = int(row.get("max", 100) or 100)
                        value = np.random.randint(low, high + 1)
                    except (ValueError, TypeError):
                        value = np.random.randint(0, 100)
                
                elif ctype == "float":
                    try:
                        low = float(row.get("min", 0.0) or 0.0)
                        high = float(row.get("max", 1.0) or 1.0)
                        value = float(np.random.uniform(low, high))
                    except (ValueError, TypeError):
                        value = float(np.random.uniform(0.0, 1.0))
            
            field_values.append(value)
        
        data[field_name] = field_values
        processed_fields.add(field_name)
    
    # Process all fields respecting dependencies
    for field_name in spec_df['name']:
        process_field(field_name)
    
    return pd.DataFrame(data)
# -----------------------------------------------------------------------------

def df_describe_text(df: pd.DataFrame) -> str:
    return df.describe().to_string()

# ----------------------------- Gradio UI ------------------------------------
def on_suggest(task_desc):                      return suggest_columns(task_desc)

def on_generate(spec_df, n_rows, filename):
    n_rows = int(n_rows)
    df     = generate_data(spec_df, n_rows)
    stats  = df_describe_text(df)
    filename = (filename or "synthetic_data").strip()
    tmp_dir  = tempfile.gettempdir()
    path     = os.path.join(tmp_dir, f"{filename}.xlsx")
    try:
        df.to_excel(path, index=False)
    except ModuleNotFoundError:
        path = os.path.join(tmp_dir, f"{filename}.csv")
        df.to_csv(path, index=False)
    return df, stats, path

with gr.Blocks(title="Random Data Generator") as demo:
    gr.Markdown("## 🧪 Random Data Generator (Llama 3.2:1B + Gradio)")
    with gr.Row():
        task_box  = gr.Textbox(label="Dataset description or columns", lines=2)
        sugg_btn  = gr.Button("Suggest Columns")

    spec_df = gr.Dataframe(headers=SPEC_COLUMNS,
                           datatype=["str"]*len(SPEC_COLUMNS),
                           label="Column specification (make sure to edit types that don't match your column name)",
                           wrap=True)

    sugg_btn.click(on_suggest, inputs=task_box, outputs=spec_df)

    with gr.Row():
        rows_in   = gr.Number(label="# Rows", value=100, precision=0)
        fname_in  = gr.Textbox(label="Output file name (no ext.)", value="synthetic_data")
        gen_btn   = gr.Button("Generate Data")

    preview_df = gr.Dataframe(label="Synthetic data preview")
    stats_box  = gr.Textbox(label="df.describe()", lines=10)
    download   = gr.File(label="Download")

    gen_btn.click(on_generate,
                  inputs=[spec_df, rows_in, fname_in],
                  outputs=[preview_df, stats_box, download])

if __name__ == "__main__":
    demo.launch(allowed_paths=[tempfile.gettempdir()])