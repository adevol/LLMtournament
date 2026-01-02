# System Architecture

## Overview
The LLM Tournament backend has been refactored into a **Service-Oriented Architecture (SOA)** to improve scalability, maintainability, and testability. It exposes a REST API via **FastAPI** for frontend integration.

## Core Components

### 1. API Layer (`src/llm_tournament/api/`)
- Built with **FastAPI**.
- Exposes endpoints for:
  - **Matches**: Retrieve match history and details.
  - **Rankings**: Get current leaderboards (Elo/TrueSkill).
- Uses dependency injection for configuration and storage.

### 2. Service Layer (`src/llm_tournament/services/`)
Encapsulates business logic into distinct domains:

- **`SubmissionService`**: 
  - Manages the creative lifecycle of essays.
  - Orchestrates **Generation** (Writer LLMs), **Critique** (Critic LLMs), and **Revision** (Writer LLMs).
  
- **`MatchService`**:
  - Manages the competitive aspect.
  - Implements **Swiss Pairing** logic to pair candidates.
    - **Pairing Logic**: Uses a robust pairing strategy:
        1.  **Strict Pass**: Attempts to find pairs who have *not* played each other yet.
        2.  **Fallback Pass**: If unpaired candidates remain, allows rematches to ensure participation.
        3.  **Bye Handling**: If an odd number of candidates exist, one is selected for a "Bye" (lowest rated with fewest byes) before pairing begins.
  - Orchestrates **Judging** (Judge LLMs) and **Auditing** (Confidence checks).
  - Updates rankings based on match results.

- **`LLMClient`**: 
  - abstract interface for LLM providers (currently OpenRouter).
  - Handles caching (DuckDB), retries, and rate limiting.

### 3. Data Layer (`src/llm_tournament/models/`, `src/llm_tournament/services/storage/`)
- **SQLModel** (SQLAlchemy + Pydantic) is used for all database entities.
- **Entities**:
  - `Match`: Stores detailed match results, including judge reasoning and transcripts.
  - `Rating`: Stores current ratings for each candidate/topic.
- **`TournamentStore`**: Unified repository pattern for database access.

## Directory Structure

```text
src/llm_tournament/
├── api/                # FastAPI application & routers
├── core/               # Configuration & Shared Utilities
├── models/             # Database Schemas (SQLModel)
├── services/           # Business Logic
│   ├── llm/            # LLM Client & Caching
│   ├── match/          # Pairing, Judging, Ranking Logic
│   └── submission/     # Generation, Critique, Revision
└── main.py             # Entry point
```
