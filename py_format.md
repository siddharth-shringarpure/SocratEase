# **General Project Standards**

## **Project Overview**

## **Architecture Standards**

### **Core Stack**

* **Backend**: FastAPI (Python 3.10+)  
* **Frontend**: React \+ TypeScript (Vite/Next.js) \+ Tailwind CSS  
* **Package Manager**: uv (backend), pnpm (frontend)  
* **Database**: \[Postgres / Neo4j / SQLite \- defined per project\]  
* **Async Processing**: \[RabbitMQ / Redis \- if applicable\]

## **Python Style Guidelines**

### **Code Formatting & Structure**

* **Line length**: Maximum 80 characters (flexible for long imports, URLs, or when auto-formatters can't help)  
* **Indentation**: 4 spaces (never tabs); use hanging indents for line continuation  
* **Imports**:  
  * One per line  
  * **Group order**: stdlib \-\> third-party \-\> local sub-packages (blank line between groups)  
  * Sort lexicographically within groups; use from x import y for modules  
  * Full package paths (eg: from absl import flags, not import flags)  
  * **Avoid from typing import ...**: Use built-in types (eg: list\[int\], dict\[str, int\]) instead of List, Dict, etc.  
  * Example structure:  
    Python  
    """Module docstring here."""

    import logging  
    from pathlib import Path

    import pandas as pd

    from .constants import COLUMN\_NAMES  
    from .base import BaseClass

* **Naming conventions**:  
  * module\_name, function\_name, local\_var\_name: lower\_with\_under  
  * ClassName, ExceptionName: CapWords  
  * GLOBAL\_CONSTANT\_NAME: CAPS\_WITH\_UNDER  
  * Internal/private: prefix with single underscore \_private\_var  
* **Strings**: Prefer f-strings or % formatting; use """ for docstrings and multi-line strings  
* **Quote style**:  
  * Use double quotes for strings and multi-character references (eg: "hello", "some\_function")  
  * Use single quotes only for single characters (eg: 'a', 'x')  
  * Examples in documentation: use "eg:" not "e.g." and "ie:" not "i.e.,"  
* **Comprehensions**: Allowed for simple cases; avoid multiple for clauses or complex filters

### **Type Annotations**

* **Always add return type hints** to all functions and methods  
  * Use \-\> None for methods that don't return values  
  * Use \-\> int, \-\> str, \-\> pd.DataFrame etc. for explicit returns  
  * Omit return hint only for \_\_init\_\_ methods (implicitly return None)  
* Use modern syntax: str | None (not Optional\[str\]) in Python 3.10+  
* Use built-in types: list\[int\], dict\[str, Any\] instead of List\[int\], Dict\[str, Any\]  
* **Avoid from typing import ...**: Python 3.9+ supports built-in generic types  
* **Avoid redundant type annotations** on local variables \- let type inference work:  
  * Bad: count: int | None \= get\_count(...)  
  * Good: count \= get\_count(...)  
* **Avoid from \_\_future\_\_ import annotations**: Not needed in Python 3.10+  
* For forward references, use string quotes (eg: def foo(self) \-\> "Bar":)  
* Import types only when needed: from typing import Any, Protocol (not common types)

### **Documentation**

* **Module docstrings**: Required at top of every module with summary  
  * Format: """Brief summary.\\n\\nDetailed description if needed.\\n"""  
  * Example: """User management module.\\n\\nHandles authentication and profile updates...\\n"""  
* **Function/method docstrings**: Document all public methods; optional for obvious private helpers  
  * Use Google style with Args:, Returns:, Raises: sections  
  * Prefer imperative mood: "Fetch rows from database" over "Fetches rows"  
  * **Keep concise** \- one-line summary, minimal Args descriptions, avoid redundancy  
  * Example:  
    Python  
    def fetch\_users(self, active\_only: bool \= True) \-\> list\[User\]:  
        """Retrieve user records from database.

        Args:  
            active\_only: Filter to active users only (default: True)

        Returns:  
            List of User objects

        Raises:  
            DatabaseError: If connection fails  
        """

  * Format with full stop at end of summary line only; no full stops in Args/Returns/Raises lines  
* **Docstring formatting rules**:  
  * Remove articles ("the", "a") where clarity isn't lost  
  * Default values in Args: use (default: value) format  
  * Keep parameter descriptions brief \- defer details to code  
  * Use Note: section sparingly for legacy/deprecated warnings  
* **Classes**: Summary \+ Attributes: section for public attributes only  
* **Comments**:  
  * Explain "why" not "what"  
  * Remove obvious comments that restate code  
  * Use \# TODO: \<context\> \- \<description\> format

### **Common Patterns**

* **Error handling**: Use specific exceptions (ValueError, ConnectionError); avoid bare except:  
  * Keep error messages concise but informative  
  * Format: f"Brief description: {details}" not multi-line explanations  
  * Example: raise ValueError(f"Invalid status '{status}'. Must be one of {VALID\_STATUSES}")  
* **Resource management**:  
  * Always use with statements (context managers) for files, sockets, and database connections.  
  * Define custom @contextmanager helpers to encapsulate complex resource lifecycles (setup/teardown) rather than using manual try...finally blocks.  
  * Ensure cleanup is guaranteed even on exceptions.  
* **Refactoring & DRY**:  
  * Encapsulate repeated logic (auth checks, response mapping, error handlers) into **private helper functions** (prefixed with \_).  
  * Prefer explicit helpers over complex decorators to maintain readability and simpler control flow.  
* **File/directory deletion**:  
  * Files (.pdf, .zip, .csv) → path.unlink() or path.unlink(missing\_ok=True)  
  * Empty directories only → path.rmdir() (fails if not empty \-- prevents accidental data loss)  
  * Directories with contents → shutil.rmtree(path) (recursive deletion)  
  * Rule of thumb: Use most restrictive method that works (unlink for files, rmdir for empty dirs, rmtree only when needed)  
* **Default args**: Never use mutable defaults (def foo(x=\[\])); use None with conditional init  
* **Truth testing**: Use implicit false (if not users: not if len(users) \== 0:)  
* **Main guard**: Always check if \_\_name\_\_ \== "\_\_main\_\_": before executing main logic  
* **Logging & Print Statements**:  
  * **Logging configuration**: Import configure\_logging from ..core and call it after imports  
    * Good: from ..core import configure\_logging then configure\_logging()  
    * Bad: logger \= logging.getLogger(\_\_name\_\_) (inconsistent with codebase)  
  * Use logging.info(...), logging.debug(...), etc. directly (not via a logger variable)  
  * **Use lazy % formatting in logging functions** (not f-strings) to avoid Pylint W1203:logging-fstring-interpolation  
  * F-strings in logging are inefficient \-- they evaluate even when log level is disabled  
  * Good: logging.info("Retrieved %d records from database", count)  
  * Good: logging.info("Processing %s for user %d", filename, user\_id)  
  * Bad: logging.info(f"Retrieved {count} records") (triggers Pylint warning)  
  * **No leading whitespace** in string literals \-- use \\t for indentation, never raw spaces  
    * Good: print(f"\\t-\> {result}") or logging.info("\\t- Item: %s", item)  
    * Bad: print(f" \-\> {result}") (raw spaces)  
  * **No leading newlines** in quotes: Bad: print("\\nMessage") \-- avoid this where possible  
  * **Do use f-strings for any print statements**  
  * **Status symbols**: Use Unicode ✓ and ✗ or ASCII alternatives (avoid emoji ✅ ❌):  
    * Good: logging.info("✓ Table created successfully")  
    * Good: logging.info("\[OK\] Table created successfully")  
    * Good: logging.info("SUCCESS: %d events processed", count)  
    * Good: logging.info("✗ Failed to connect")  
    * Bad: logging.info("✅ Done") or logging.info("❌ Failed")  
  * **Dividers**: ALWAYS import DIVIDER from constants.py \-- never use "=" \* 60 or similar inline  
    * Good: from ..core.constants import DIVIDER then print(DIVIDER)  
    * Bad: print("=" \* 60\) or print("\\n" \+ "=" \* 60\)

### **Punctuation & Typography**

* **Arrows**: Use ASCII \-\> not Unicode →  
  * Good: Input \-\> Process \-\> Output  
  * Bad: Input → Process → Output  
* **Dashes**:  
  * **Hyphen** (-): Compound words, line breaks (eg: "multi-step", "well-known")  
  * **En dash** (--): Ranges, no spaces (eg: "2020--2025", "pages 10--15")  
  * **Em dash** (--): Breaks in thought, must have spaces on either side (eg: "The results \-- as expected \-- were positive")  
  * Use en dash for numeric/date ranges: "January--March", "10--20 items"  
  * Use em dash sparingly for parenthetical statements  
* **Symbols**: Prefer ASCII over Unicode where possible  
  * Use \>=, \<=, \!= not ≥, ≤, ≠  
  * Use \~ not ≈ for "approximately"  
  * Use x or \* not × for multiplication

### **Refactoring Best Practices**

* **DRY principle**: Extract repeated patterns into helper methods  
  * Name helpers with leading underscore for private: \_create\_table(), \_validate\_data()  
  * Keep helpers simple and focused on one task  
* **KISS principle**: Favor simple solutions over clever ones  
  * Avoid regex when simple string methods work  
  * Don't add features "just in case" \- implement when needed  
* **Separation of concerns**:  
  * Helper methods do work, callers handle logging/reporting  
  * Validation logic separate from business logic  
  * SQL queries separate from execution logic (consider extracting to constants)  
* **Type inference**: Let Python infer local variable types \- only annotate parameters and returns

### **Project-Specific Adaptations**

* **Fish shell**: Be aware of fish shell limitations (no heredocs) when writing scripts  
* **Package manager**: Use uv for backend folder; system Python \+ pip for root  
* **Linting**: Run pylint on code; suppress with \# pylint: disable=invalid-name and explanation

## **Critical Conventions**

### **File Paths & Directory Structure**

* **Never hardcode paths** \- always reference backend/core/constants.py for project paths  
* Use constants: PROJECT\_ROOT, DATA\_DIR, RESOURCES\_DIR, LOGS\_DIR, etc.  
* Data directories created lazily: output\_dir.mkdir(parents=True, exist\_ok=True)

### **Backend Folder Structure & Naming Conventions**

The backend follows a **layered architecture** pattern to separate concerns and maintain scalability:

#### **Directory Structure**

backend/  
├── routers/           \# HTTP layer (FastAPI endpoints)  
│   ├── auth\_router.py  
│   ├── items\_router.py  
│   └── users\_router.py  
├── services/          \# Business logic layer  
│   ├── auth\_service.py  
│   ├── item\_service.py  
│   ├── file\_service.py  
│   └── processing\_service.py  
├── db/                \# Data access layer (DB wrappers/Repositories)  
│   ├── auth.py        \# AuthDB class  
│   ├── items.py       \# ItemDB class  
│   └── base.py        \# DBManager base class  
├── workers/           \# Background process entry points  
│   └── scheduler\_worker.py  
├── scripts/           \# One-off utilities (migrations, seeding)  
└── core/              \# Shared constants, models, config

#### **Layer Responsibilities**

1. **Routers (routers/)** — "The Traffic Cop"  
   * **Purpose**: Handle HTTP requests/responses only  
   * **Naming**: Always use \_router.py suffix (eg: auth\_router.py, items\_router.py)  
   * **Rules**:  
     * No business logic — delegate to services  
     * No direct database calls — use services  
     * Handle request validation (Pydantic models)  
     * Map exceptions to HTTP status codes  
   * **Example**: auth\_router.py validates login request, calls AuthService, returns token  
2. **Services (services/)** — "The Brains"  
   * **Purpose**: Contain all business logic and orchestration  
   * **Naming**: Always use \_service.py suffix (eg: item\_service.py, auth\_service.py)  
   * **Rules**:  
     * Stateless operations (accept state as parameters)  
     * Call db/ layer for data operations  
     * Raise domain exceptions (not HTTP exceptions)  
     * Reusable across routers, workers, and scripts  
   * **Example**: processing\_service.py orchestrates data transformation and validation  
3. **Database Layer (db/)** — "The Librarian"  
   * **Purpose**: Encapsulate all database operations (SQL, Graph, or NoSQL)  
   * **Naming**: Use descriptive names without suffix (eg: auth.py, items.py)  
   * **Rules**:  
     * No business logic — only CRUD operations  
     * Use context managers (with statements)  
     * Return raw data (dicts, lists) or simple DTOs, not complex domain objects  
   * **Example**: AuthDB handles user creation and session management  
4. **Workers (workers/)** — "The Schedulers"  
   * **Purpose**: Entry points for background processes (cron jobs, long-running tasks)  
   * **Naming**: Use \_worker.py suffix (eg: scheduler\_worker.py)  
   * **Rules**:  
     * Minimal logic — import and call services  
     * Include main() function with if \_\_name\_\_ \== "\_\_main\_\_" guard  
     * Handle CLI arguments and orchestration only  
5. **Scripts (scripts/)** — "The Utilities"  
   * **Purpose**: One-off maintenance tasks (migrations, seeding, setup)  
   * **Rules**:  
     * Not part of the running application  
     * Run manually via python \-m backend.scripts.script\_name  
     * Can import from services/db as needed

#### **Naming Anti-Patterns to Avoid**

* **Don't use generic suffixes**:  
  * Bad: utils.py, helpers.py, common.py  
  * Good: file\_service.py, auth\_service.py (explicit purpose)  
* **Don't mix concerns in one file**:  
  * Bad: Service class \+ worker main() in same file  
  * Good: Service in services/, worker in workers/ imports service  
* **Don't create "mega-services"**:  
  * Bad: general\_service.py with 20+ unrelated functions  
  * Good: Domain-specific services (user\_service.py, billing\_service.py)

#### **Import Patterns**

Python

\# In routers/auth\_router.py  
from ..services.auth\_service import AuthService  
from ..db.auth import AuthDB

\# In services/data\_service.py  
from ..services.file\_service import FileService  
from ..pipelines import IngestionPipeline

## **Development Workflow**

### **Running Scripts**

* **Backend (FastAPI)**: From project root, run uv run fastapi dev backend/main.py  
  * Server: http://localhost:8000  
  * API docs: http://localhost:8000/docs (Swagger UI)  
* **Frontend (React)**: cd frontend && pnpm i; pnpm dev  
  * App: http://localhost:3000  
* **Individual scripts**: Run as modules from project root: uv run python \-m backend.module.file  
  * Example: uv run python \-m backend.scripts.seed\_db  
  * Always use module syntax (no .py extension)  
  * Execute from project/ directory (the Python package root)  
* Fish shell used (no heredocs support \- use printf/echo for multiline)

### **Testing Workflow**

* **Unit/Integration**: Use pytest for formal testing  
* **Ad-hoc**: Notebooks allowed for exploratory analysis (notebooks/ folder)  
* **Console Feedback**: Include flush=True in print statements for real-time feedback during long operations

### **Common Tasks**

* **Add new constants**: Define in backend/core/constants.py (paths, URLs, enum classes)  
* **Database operations**: Use classes in backend/db/  
* **Data models**: Define in backend/core/data\_models.py  
* **API endpoints**: Add to backend/main.py (FastAPI routes)

## **Anti-Patterns to Avoid**

### **General Code Quality**

* **Don't hardcode paths** \- always derive from Path(\_\_file\_\_).parent or use constants.py  
* **Don't use redundant type annotations** \- avoid query: str \= "SELECT..." (inference works)  
* **Don't over-comment** \- remove comments that restate obvious code  
* **Don't overuse decorative logging** \- use ✓ only for important completion milestones  
* **Don't leave dead code** \- remove unused functions, commented code, or legacy methods  
* **Don't add unnecessary complexity**:  
  * No regex for simple string operations  
  * No helper parameters that can be inferred from the data  
  * No "future-proofing" features not currently needed

### **Data & Logic**

* **Don't hardcode URLs** \- use constants from constants.py  
* **Don't hardcode DB queries/labels/types** \- always use Enum classes in constants.py (eg: NodeLabel, RelationType, TableName)  
* **Don't assume data completeness** \- always handle NaN or None  
* **Don't mix data validation with business logic** \- extract validators as separate methods  
* **ALWAYS USE BRITISH SPELLING** for comments, docstrings, and user-facing text (eg: "favour", "organisation")

## **Python Import Rules (Critical)**

### **Package Structure**

* project/ is the Python package root (workspace root in VS Code)  
* Code executed with python \-m backend.module.file (always from project root)  
* backend/ is a proper Python package with \_\_init\_\_.py files

### **Import Guidelines**

**ALWAYS USE relative imports inside backend/ for siblings:**

Python

from ..core.constants import PROJECT\_ROOT, DB\_DIR  
from ..core.data\_models import UserMetadata  
from ..services.data\_service import DataService  
from ..db.items import ItemDB

**For convenience re-exports from \_\_init\_\_.py:**

Python

from ..core import configure\_logging  \# OK if exported in \_\_init\_\_.py

**Absolute imports allowed but discouraged:**

Python

from backend.core.constants import PROJECT\_ROOT  \# Works but avoid

**FORBIDDEN \- Never use these:**

* sys.path.append(...) or sys.path.insert(...)  
* PYTHONPATH environment variable  
* Running files directly: python file.py  
* Top-level sibling imports: from core import X  
* Wildcard barrel imports: from .module import \* (without explicit \_\_all\_\_)  
* Auto-aggregating all submodules in \_\_init\_\_.py

### **Circular Import Prevention**

* **Be vigilant about circular dependencies** between modules  
* If module A needs B and B needs A, refactor shared logic to a third module  
* Use string quotes for forward references in type annotations (eg: \-\> "ClassName")  
* Import types inside functions if needed only for type checking  
* Structure: constants.py \-\> data\_models.py \-\> db/ \-\> services/ \-\> main.py

### **\_\_init\_\_.py Rules**

* Should be minimal and deliberate  
* **Allowed**: Selective explicit re-exports for public API convenience  
  * Example: from .module import specific\_function with \_\_all\_\_ list  
* **Allowed**: Wildcard re-exports for constants/models modules  
  * Example: from .constants import \* \# noqa: F403  
  * Use sparingly \- only for modules with many constants/models  
  * Must use \# noqa: F403 to acknowledge intentional wildcard  
* **Forbidden**: Wildcard imports that re-export everything  
  * Never: from .module import \* for business logic modules  
  * Never: Automatically aggregating all submodules without intent  
* Use \_\_all\_\_ to explicitly declare public API surface

### **Tooling Configuration**

* VS Code workspace root: project/  
* All imports work identically for CLI, FastAPI, tests, and CI  
* Linter false-positives fixed via tooling config, never code changes  
* No environment-dependent import behavior

## **Database Naming & Interaction Conventions**

* **Tables/Nodes**: Use CapWords or PascalCase (defined in constants)  
* **Columns/Properties**: Use camelCase for Graph DBs, snake\_case for SQL (unless strictly defined otherwise)  
* **Relationships**: Use CAPS\_WITH\_UNDER format (defined in constants.RelationType class)

### **Driver Lifecycle Patterns**

**Two distinct patterns for Database driver usage:**

1. **Standalone scripts** (use ephemeral connections):  
   * Creates new connection/driver per call  
   * Automatically closes connection after execution  
   * Suitable for one-off queries in scripts, setup tasks, or simple operations  
2. **Service classes** (use injected driver instance):  
   * Services accept a Driver/Session instance in constructor (eg: UserService(driver))  
   * Driver lifecycle managed by caller (typically FastAPI lifespan or pipeline context)  
   * Efficient for multiple operations within same workflow  
   * Files: services/user\_service.py, services/data\_service.py

**Do not refactor service classes to use ephemeral connections** \- the per-call creation/teardown is inefficient for high-throughput operations.
