import os
import json
import argparse
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import shutil
from fastapi.responses import JSONResponse
from rapidfuzz import fuzz
import re

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("model"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
)

def local_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def extract_text_from_pdf(file_path: str, verbose=False) -> str:
    if verbose:
        print("\U0001F4E5 Ingesting PDF...")
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    if verbose:
        print("\n\U0001F4C4 Raw PDF Text:\n", text)
    return text.strip()

class AgentState(TypedDict):
    file_path: str
    input_type: str
    pdf_content: str
    preprocessed_text: str
    summary: str
    nfr_rules: List[Dict]
    compliance_results: List[Dict]
    remediation_actions: List[str]
    next_node: str
    policy_content: str


# --- Nodes ---
def pdf_ingestion_node(state: AgentState):
    text = extract_text_from_pdf(state["file_path"])
    return {**state, "pdf_content": text, "next_node": "decide_branch"}

def decide_branch_node(state: AgentState):
    if state["input_type"].lower() == "raw":
        return {**state, "next_node": "preprocess"}
    else:
        return {**state, "preprocessed_text": state["pdf_content"], "next_node": "summarize"}

def preprocess_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a PDF Preprocessor. Structure into valid JSON."),
        HumanMessage(content=state["pdf_content"])
    ])
    response = llm.invoke(prompt.format_messages())
    try:
        parsed = json.loads(response.content)
        structured = json.dumps(parsed, indent=2)
        return {**state, "preprocessed_text": structured, "next_node": "summarize"}
    except json.JSONDecodeError:
        return {**state, "preprocessed_text": response.content, "next_node": "summarize"}

def summarize_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a Summarizer."),
        HumanMessage(content=f"Summarize:\n{state['preprocessed_text']}")
    ])
    response = llm.invoke(prompt.format_messages())
    return {**state, "summary": response.content, "next_node": "generate_nfr_rules"}

def generate_nfr_rules_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a system design and compliance expert.

Your task is to generate **non-functional requirement (NFR) rules** that are:
- Specific to the given BRD summary,
- Measurable and actionable,
- Informed by the IT policy reference,
- Written in original wording (do not copy or restate IT policy verbatim),
- Strictly aligned with standards or constraints implied by the IT policy.

Avoid vague or generic NFRs. Only include rules that make logical sense **within the business context of the BRD** and **informed by the policy's expectations**. Each rule must be testable and relevant.
"""),
        HumanMessage(content=f"""BRD Summary:
{state['summary']}

IT Policy Reference:
{state['policy_content']}

Generate a list of tailored NFR rules (one per line):""")
    ])
    response = llm.invoke(prompt.format_messages())
    rules = [{"rule": r.strip(), "status": "pending"} for r in response.content.split("\n") if r.strip()]
    return {**state, "nfr_rules": rules, "next_node": "validate_compliance"}



def validate_compliance_node(state: AgentState, threshold: int = 60):
    try:
        df = pd.read_excel(local_path("presaved_rules.xlsx"))
        if "Rules" not in df.columns:
            return {**state, "compliance_results": [], "next_node": "remediate"}
    except Exception:
        return {**state, "compliance_results": [], "next_node": "remediate"}

    df = df.dropna(subset=["Rules"])
    presaved_rules = df["Rules"].str.strip().str.lower().tolist()
    used = set()
    results = []

    for r in state["nfr_rules"]:
        rt = r["rule"].strip().lower()
        best_score = 0
        best_idx = -1
        match = None
        for i, rule in enumerate(presaved_rules):
            if i in used:
                continue
            score = fuzz.partial_ratio(rt, rule)
            if score > best_score:
                best_score, best_idx, match = score, i, rule
        if best_score >= threshold:
            used.add(best_idx)
            results.append({"rule": r["rule"], "status": "passed", "matched_with": match, "score": f"{best_score:.2f}%"})
        else:
            results.append({"rule": r["rule"], "status": "failed"})

    return {**state, "compliance_results": results, "next_node": "remediate"}   

def remediate_node(state: AgentState):
    failed = [
        r["rule"] for r in state["compliance_results"]
        if r["status"] == "failed" and not r.get("matched_with")
    ]
    actions = []
    for rule in failed:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Compliance Remediation Expert. Provide a 2-line fix:
- First line: Issue:
- Second line: Fix:"""),
            HumanMessage(content=rule)
        ])
        response = llm.invoke(prompt.format_messages())
        actions.append(response.content.strip())
    return {**state, "remediation_actions": actions, "next_node": END}

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("pdf_ingestion", pdf_ingestion_node)
graph.add_node("decide_branch", decide_branch_node)
graph.add_node("preprocess", preprocess_node)
graph.add_node("summarize", summarize_node)
graph.add_node("generate_nfr_rules", generate_nfr_rules_node)
graph.add_node("validate_compliance", validate_compliance_node)
graph.add_node("remediate", remediate_node)

graph.add_edge("pdf_ingestion", "decide_branch")
graph.add_conditional_edges("decide_branch", lambda s: s["next_node"], {
    "preprocess": "preprocess",
    "summarize": "summarize"
})
graph.add_edge("preprocess", "summarize")
graph.add_edge("summarize", "generate_nfr_rules")
graph.add_edge("generate_nfr_rules", "validate_compliance")
graph.add_edge("validate_compliance", "remediate")
graph.add_edge("remediate", END)
graph.set_entry_point("pdf_ingestion")
app = graph.compile()

# --- Pipeline Runner ---
def run_full_pipeline(pdf_path: str, input_type="raw", policy_text=""):
    text = extract_text_from_pdf(pdf_path, verbose=True)
    state = {
        "file_path": pdf_path,
        "input_type": input_type,
        "pdf_content": text,
        "policy_content": policy_text,
        "preprocessed_text": "",
        "summary": "",
        "nfr_rules": [],
        "compliance_results": [],
        "remediation_actions": [],
        "next_node": "pdf_ingestion"
    }
    result = app.invoke(state)

    def clean(text):
        text = text.replace("**", "").replace("→", "").replace("-", "")
        text = re.sub(r"#.*", "", text)
        text = re.sub(r':[a-z_]+:', '', text)
        return text.strip()

    summary = clean(result.get("summary", ""))
    nfr = [clean(r["rule"]) for r in result.get("nfr_rules", []) if "not found" not in r["rule"].lower()]
    comp = []
    for r in result.get("compliance_results", []):
        if "not found" in r["rule"].lower(): continue
        d = {"rule": clean(r["rule"]), "status": r["status"]}
        if r.get("matched_with"):
            d["matched_with"] = clean(r["matched_with"])
            d["score"] = r["score"]
        comp.append(d)
    remed = [clean(r) for r in result.get("remediation_actions", []) if "not found" not in r.lower()]

    print("\nSUMMARY:\n", summary)
    print("\nNFR RULES:")
    [print(r) for r in nfr]

    print("\nCOMPLIANCE RESULTS:")
    for r in comp:
        print(f"Rule: {r['rule']}\nStatus: {r['status']}")
        if r.get("matched_with"):
            print(f"Matched With: {r['matched_with']} (Score: {r['score']})")


    print("\nREMEDIATION SUGGESTIONS:")
    [print(r) for r in remed]



    return {"summary": summary, "nfr_rules": nfr, "compliance_results": comp, "remediation_actions": remed}

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run-all", "preprocess", "summarize", "generate_nfr_rules", "validate_compliance", "remediate"], required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--type", default="raw")
    parser.add_argument("--policy", required=False, help="Path to policy document")

    args = parser.parse_args()

    verbose = args.mode == "run-all"
    text = extract_text_from_pdf(args.file, verbose=verbose)
    policy_text = extract_text_from_pdf(args.policy) if args.policy else ""
    state = {
        "file_path": args.file,
        "input_type": args.type,
        "pdf_content": text,
        "policy_content": policy_text,
        "preprocessed_text": "",
        "summary": "",
        "nfr_rules": [],
        "compliance_results": [],
        "remediation_actions": [],
        "next_node": args.mode
    }

    if args.mode == "run-all":
        run_full_pipeline(args.file, args.type)
    else:
        node_map = {
            "preprocess": preprocess_node,
            "summarize": summarize_node,
            "generate_nfr_rules": generate_nfr_rules_node,
            "validate_compliance": validate_compliance_node,
            "remediate": remediate_node,
        }

        step_order = ["preprocess", "summarize", "generate_nfr_rules", "validate_compliance", "remediate"]
        current_state = state

        for step in step_order:
            if step == args.mode:
                result = node_map[step](current_state)

                response_data = {
                    "summary": result.get("summary", "") or "",
                    "nfr_rules": [r for r in result.get("nfr_rules", []) if isinstance(r, str) or isinstance(r, dict)],
                    "compliance_results": [
                        r for r in result.get("compliance_results", [])
                        if isinstance(r, dict) and r.get("rule") and r.get("status")
                    ],
                    "remediation_actions": [
                        r for r in result.get("remediation_actions", [])
                        if isinstance(r, str) and r.strip()
                    ],
                }

                print(json.dumps(response_data, indent=2))
                break

            current_state = node_map[step](current_state)

# --- FastAPI ---
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.post("/run_pipeline")
async def run_pipeline_endpoint(
    file: UploadFile = File(...),
    input_type: str = Form("raw"),
    policy: UploadFile = File(None)
):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)

    policy_text = ""
    if policy:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as pol_tmp:
            shutil.copyfileobj(policy.file, pol_tmp)
            policy_text = extract_text_from_pdf(pol_tmp.name)

    result = run_full_pipeline(tmp.name, input_type, policy_text)
    return JSONResponse(content=result)


@api.post("/run_step")
async def run_step_endpoint(
    file: UploadFile = File(...),
    input_type: str = Form("raw"),
    step: str = Form(...),
    policy: UploadFile = File(None)
):
    # Save BRD PDF
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        brd_path = tmp.name

    # Extract BRD content
    text = extract_text_from_pdf(brd_path)

    # Extract policy content if provided
    policy_text = ""
    if policy:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as pol_tmp:
            shutil.copyfileobj(policy.file, pol_tmp)
            policy_text = extract_text_from_pdf(pol_tmp.name)

    # Initialize agent state
    state = {
        "file_path": brd_path,
        "input_type": input_type,
        "pdf_content": text,
        "policy_content": policy_text,
        "preprocessed_text": "",
        "summary": "",
        "nfr_rules": [],
        "compliance_results": [],
        "remediation_actions": [],
        "next_node": step
    }

    node_map = {
        "preprocess": preprocess_node,
        "summarize": summarize_node,
        "generate_nfr_rules": generate_nfr_rules_node,
        "validate_compliance": validate_compliance_node,
        "remediate": remediate_node,
    }

    step_order = ["preprocess", "summarize", "generate_nfr_rules", "validate_compliance", "remediate"]

    def clean(text):
        text = text.replace("**", "").replace("→", "").replace("-", "")
        text = re.sub(r"#.*", "", text)
        text = re.sub(r':[a-z_]+:', '', text)
        return text.strip()

    current_state = state
    result = {}

    for s in step_order:
        current_state = node_map[s](current_state)
        if s == step:
            if s == "summarize":
                result = {"summary": clean(current_state.get("summary", ""))}
            elif s == "generate_nfr_rules":
                result = {
                    "nfr_rules": [
                        clean(r["rule"]) for r in current_state.get("nfr_rules", []) if isinstance(r, dict)
                    ]
                }
            elif s == "validate_compliance":
                result = {
                    "compliance_results": [
                        {
                            "rule": clean(r["rule"]),
                            "status": r["status"],
                            **({"matched_with": clean(r["matched_with"]), "score": r["score"]} if "matched_with" in r else {})
                        }
                        for r in current_state.get("compliance_results", []) if isinstance(r, dict)
                    ]
                }
            elif s == "remediate":
                result = {
                    "remediation_actions": [
                        clean(r) for r in current_state.get("remediation_actions", []) if isinstance(r, str)
                    ]
                }
            break

    return JSONResponse(content=result)
