# Neo4J Aura — Connection Setup

## 1. Create a Free AuraDB Instance

1. Go to [https://neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/) and sign up or log in.
2. Click **"New Instance"**.
3. Select **AuraDB Free** (the free tier is sufficient for this project).
4. Choose a name for the instance (e.g. `graph-rag`) and click **Create**.

---

## 2. Save Your Credentials (One-Time Only)

Immediately after the instance is created, Neo4J will display a credentials panel. **This is the only time the password is shown in plain text.**

The panel contains three values:

| Field | Example value | .env key |
|---|---|---|
| Connection URI | `neo4j+s://xxxxxxxx.databases.neo4j.io` | `NEO4J_URI` |
| Username | `neo4j` | `NEO4J_USERNAME` |
| Password | `AbCdEf1234...` (auto-generated) | `NEO4J_PASSWORD` |

Click **"Download credentials"** to save a `.txt` file, or copy each value manually.

> If you close this panel without saving, you will need to reset the password from the instance dashboard (Actions → Reset Password).

---

## 3. Wait for the Instance to Start

The instance status will show **"Creating"** for about 1–2 minutes. Wait until it turns **"Running"** before connecting.

---

## 4. Fill in the `.env` File

Open the `.env` file in the project root and add the three values:

```
OPENAI_API_KEY=<your key>

NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your auto-generated password>
```

> The `neo4j+s://` scheme is required for Aura's TLS-encrypted connection. Do not use `bolt://` or `neo4j://`.

---

## 5. Verify the Connection

You can verify the connection is working before running the notebooks by opening the Aura console and clicking **"Open"** → **"Query"**. Run:

```cypher
MATCH (n) RETURN count(n)
```

If it returns `0` (empty database), the connection is healthy and the project is ready to run.

---

## Running Order

Always run the notebooks in this order:

1. **`01_ingestion.ipynb`** — loads the PDF, extracts the knowledge graph, and stores it in Neo4J.
2. **`02_retrieval.ipynb`** — connects to the populated graph and runs queries.

Running `02_retrieval.ipynb` before `01_ingestion.ipynb` will result in an empty graph and no useful answers.
