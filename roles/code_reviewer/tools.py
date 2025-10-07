"""
Code Reviewer Custom Tools

Specialized tools for code analysis and review tasks.
"""

import ast
import logging
import re
from datetime import datetime
from typing import Dict, List

from strands import tool

logger = logging.getLogger(__name__)


@tool
def analyze_code_quality(code: str, language: str = "python") -> Dict:
    """
    Analyze code quality and identify potential issues.

    Args:
        code: Source code to analyze
        language: Programming language (python, javascript, java, etc.)

    Returns:
        Dict containing code quality analysis results
    """
    try:
        issues = []
        suggestions = []
        metrics = {}

        if language.lower() == "python":
            issues, suggestions, metrics = _analyze_python_code(code)
        elif language.lower() in ["javascript", "js"]:
            issues, suggestions, metrics = _analyze_javascript_code(code)
        else:
            # Generic analysis for other languages
            issues, suggestions, metrics = _analyze_generic_code(code)

        # Calculate overall quality score
        quality_score = max(0, 100 - (len(issues) * 10) - (len(suggestions) * 5))

        return {
            "success": True,
            "language": language,
            "quality_score": quality_score,
            "issues": issues,
            "suggestions": suggestions,
            "metrics": metrics,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Code quality analysis failed: {e}")
        return {"success": False, "error": str(e), "language": language}


def _analyze_python_code(code: str) -> tuple:
    """Analyze Python code specifically."""
    issues = []
    suggestions = []
    metrics = {}

    lines = code.split("\n")
    metrics["line_count"] = len(lines)
    metrics["non_empty_lines"] = len([line for line in lines if line.strip()])

    # Check for common Python issues
    if "import *" in code:
        issues.append("Avoid wildcard imports (import *)")

    if re.search(r"except:", code):
        issues.append("Bare except clauses should specify exception types")

    if len([line for line in lines if len(line) > 100]) > 0:
        suggestions.append("Consider breaking long lines (>100 characters)")

    # Count functions and classes
    try:
        tree = ast.parse(code)
        functions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        metrics["function_count"] = len(functions)
        metrics["class_count"] = len(classes)

        # Check for docstrings
        undocumented_functions = [f for f in functions if not ast.get_docstring(f)]
        if undocumented_functions:
            suggestions.append(
                f"{len(undocumented_functions)} functions lack docstrings"
            )

    except SyntaxError:
        issues.append("Syntax error in Python code")

    return issues, suggestions, metrics


def _analyze_javascript_code(code: str) -> tuple:
    """Analyze JavaScript code specifically."""
    issues = []
    suggestions = []
    metrics = {}

    lines = code.split("\n")
    metrics["line_count"] = len(lines)
    metrics["non_empty_lines"] = len([line for line in lines if line.strip()])

    # Check for common JavaScript issues
    if "var " in code:
        suggestions.append("Consider using 'let' or 'const' instead of 'var'")

    if "==" in code and "===" not in code:
        suggestions.append("Use strict equality (===) instead of loose equality (==)")

    if re.search(r"console\.log", code):
        suggestions.append("Remove console.log statements before production")

    # Count functions
    function_matches = re.findall(
        r"function\s+\w+|const\s+\w+\s*=\s*\(.*?\)\s*=>", code
    )
    metrics["function_count"] = len(function_matches)

    return issues, suggestions, metrics


def _analyze_generic_code(code: str) -> tuple:
    """Generic code analysis for any language."""
    issues = []
    suggestions = []
    metrics = {}

    lines = code.split("\n")
    metrics["line_count"] = len(lines)
    metrics["non_empty_lines"] = len([line for line in lines if line.strip()])

    # Generic checks
    if len([line for line in lines if len(line) > 120]) > 0:
        suggestions.append("Consider breaking very long lines (>120 characters)")

    # Check for TODO/FIXME comments
    todo_count = len(re.findall(r"TODO|FIXME|HACK", code, re.IGNORECASE))
    if todo_count > 0:
        suggestions.append(f"Found {todo_count} TODO/FIXME comments to address")

    metrics["todo_count"] = todo_count

    return issues, suggestions, metrics


@tool
def security_scan(code: str, language: str = "python") -> Dict:
    """
    Scan code for potential security vulnerabilities.

    Args:
        code: Source code to scan
        language: Programming language

    Returns:
        Dict containing security scan results
    """
    try:
        vulnerabilities = []
        warnings = []

        if language.lower() == "python":
            vulnerabilities, warnings = _scan_python_security(code)
        elif language.lower() in ["javascript", "js"]:
            vulnerabilities, warnings = _scan_javascript_security(code)
        else:
            vulnerabilities, warnings = _scan_generic_security(code)

        # Calculate security score
        security_score = max(0, 100 - (len(vulnerabilities) * 20) - (len(warnings) * 5))

        return {
            "success": True,
            "language": language,
            "security_score": security_score,
            "vulnerabilities": vulnerabilities,
            "warnings": warnings,
            "scan_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        return {"success": False, "error": str(e), "language": language}


def _scan_python_security(code: str) -> tuple:
    """Scan Python code for security issues."""
    vulnerabilities = []
    warnings = []

    # Check for dangerous functions
    if "eval(" in code:
        vulnerabilities.append("Use of eval() can lead to code injection")

    if "exec(" in code:
        vulnerabilities.append("Use of exec() can lead to code injection")

    if "input(" in code:
        warnings.append("input() should be validated to prevent injection")

    if re.search(r"subprocess\.call|os\.system", code):
        vulnerabilities.append("Shell command execution without proper sanitization")

    if "pickle.loads" in code:
        vulnerabilities.append("pickle.loads() can execute arbitrary code")

    return vulnerabilities, warnings


def _scan_javascript_security(code: str) -> tuple:
    """Scan JavaScript code for security issues."""
    vulnerabilities = []
    warnings = []

    if "eval(" in code:
        vulnerabilities.append("Use of eval() can lead to code injection")

    if "innerHTML" in code:
        warnings.append("innerHTML usage may lead to XSS vulnerabilities")

    if re.search(r"document\.write", code):
        warnings.append("document.write() can be exploited for XSS")

    return vulnerabilities, warnings


def _scan_generic_security(code: str) -> tuple:
    """Generic security scan for any language."""
    vulnerabilities = []
    warnings = []

    # Check for hardcoded credentials
    if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        vulnerabilities.append("Hardcoded password detected")

    if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        vulnerabilities.append("Hardcoded API key detected")

    return vulnerabilities, warnings


@tool
def generate_code_documentation(
    code: str, language: str = "python", doc_style: str = "google"
) -> Dict:
    """
    Generate documentation for code.

    Args:
        code: Source code to document
        language: Programming language
        doc_style: Documentation style (google, sphinx, numpy)

    Returns:
        Dict containing generated documentation
    """
    try:
        if language.lower() == "python":
            documentation = _generate_python_docs(code, doc_style)
        else:
            documentation = _generate_generic_docs(code, language)

        return {
            "success": True,
            "language": language,
            "doc_style": doc_style,
            "documentation": documentation,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        return {"success": False, "error": str(e), "language": language}


def _generate_python_docs(code: str, doc_style: str) -> str:
    """Generate Python documentation."""
    try:
        tree = ast.parse(code)
        docs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_doc = f"## Function: {node.name}\n"

                # Get existing docstring or generate placeholder
                docstring = ast.get_docstring(node)
                if docstring:
                    func_doc += f"**Description:** {docstring}\n"
                else:
                    func_doc += (
                        f"**Description:** *Documentation needed for {node.name}*\n"
                    )

                # Add parameters
                if node.args.args:
                    func_doc += "**Parameters:**\n"
                    for arg in node.args.args:
                        func_doc += f"- `{arg.arg}`: *Type and description needed*\n"

                docs.append(func_doc)

            elif isinstance(node, ast.ClassDef):
                class_doc = f"## Class: {node.name}\n"
                docstring = ast.get_docstring(node)
                if docstring:
                    class_doc += f"**Description:** {docstring}\n"
                else:
                    class_doc += (
                        f"**Description:** *Documentation needed for {node.name}*\n"
                    )

                docs.append(class_doc)

        return "\n".join(docs) if docs else "No functions or classes found to document."

    except SyntaxError:
        return "Cannot generate documentation due to syntax errors in code."


def _generate_generic_docs(code: str, language: str) -> str:
    """Generate generic documentation for any language."""
    lines = code.split("\n")

    doc = f"# Code Documentation ({language})\n\n"
    doc += f"**Total Lines:** {len(lines)}\n"
    doc += f"**Non-empty Lines:** {len([line for line in lines if line.strip()])}\n\n"
    doc += "**Analysis:** This code requires manual documentation review.\n"

    return doc


@tool
def refactor_suggestions(
    code: str, language: str = "python", focus_area: str = "general"
) -> Dict:
    """
    Provide refactoring suggestions for code improvement.

    Args:
        code: Source code to analyze
        language: Programming language
        focus_area: Focus area (general, performance, readability, maintainability)

    Returns:
        Dict containing refactoring suggestions
    """
    try:
        suggestions = []

        if language.lower() == "python":
            suggestions = _python_refactor_suggestions(code, focus_area)
        else:
            suggestions = _generic_refactor_suggestions(code, focus_area)

        return {
            "success": True,
            "language": language,
            "focus_area": focus_area,
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Refactoring analysis failed: {e}")
        return {"success": False, "error": str(e), "language": language}


def _python_refactor_suggestions(code: str, focus_area: str) -> List[str]:
    """Generate Python-specific refactoring suggestions."""
    suggestions = []

    # General suggestions
    if focus_area in ["general", "readability"]:
        if len(code.split("\n")) > 50:
            suggestions.append("Consider breaking large files into smaller modules")

        if re.search(r"def \w+\([^)]*\):.*\n(.*\n){20,}", code):
            suggestions.append(
                "Some functions appear long - consider breaking into smaller functions"
            )

    # Performance suggestions
    if focus_area in ["general", "performance"]:
        if "+=" in code and "str" in code:
            suggestions.append(
                "Consider using join() for string concatenation in loops"
            )

        if re.search(r"for.*in.*range\(len\(", code):
            suggestions.append("Consider using enumerate() instead of range(len())")

    # Maintainability suggestions
    if focus_area in ["general", "maintainability"]:
        if code.count("if") > 5:
            suggestions.append(
                "Consider using polymorphism or strategy pattern for complex conditionals"
            )

        if re.search(r"[A-Z_]{3,}", code):
            suggestions.append("Consider using configuration files for constants")

    return suggestions


def _generic_refactor_suggestions(code: str, focus_area: str) -> List[str]:
    """Generate generic refactoring suggestions."""
    suggestions = []

    lines = code.split("\n")

    if len(lines) > 100:
        suggestions.append(
            "Consider breaking large files into smaller, focused modules"
        )

    if len([line for line in lines if len(line) > 100]) > 5:
        suggestions.append(
            "Multiple long lines detected - consider improving readability"
        )

    # Check for code duplication (simple heuristic)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    duplicates = [line for line, count in line_counts.items() if count > 2]
    if duplicates:
        suggestions.append(
            f"Potential code duplication detected in {len(duplicates)} lines"
        )

    return suggestions
