# How to Login to Qdrant Cloud (Free Tier) and Get Your Credentials

## Step 1: Create an Account

1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io)
2. Click **Sign Up** (or **Log In** if you already have an account)
3. You can sign up with your email or use Google/GitHub login

---

## Step 2: Create a Free Cluster

1. After logging in, you'll land on the **Clusters** dashboard
2. Click **Create Cluster**
3. Choose the **Free** tier (labeled "Free Forever" — 1 cluster, 1GB RAM)
4. Pick a cloud provider and region closest to you (e.g., AWS us-east-1)
5. Give your cluster a name (e.g., `my-rag-cluster`)
6. Click **Create**

> The cluster takes about 1–2 minutes to spin up. You'll see its status change to **Running**.

---

## Step 3: Find Your Endpoint URL

1. On the **Clusters** page, click on your cluster name
2. You'll see a **Cluster Overview** panel
3. Look for the field labeled **Endpoint** or **URL** — it looks like:
   ```
   https://xxxx-xxxx-xxxx.us-east-1-0.aws.cloud.qdrant.io:6333
   ```
4. Copy this URL — this is your `QDRANT_URL`

---

## Step 4: Get Your API Key

1. On the same cluster overview page, click the **API Keys** tab (or look for a **Data Access Control** section)
2. Click **Create API Key**
3. Give it a name (e.g., `rag-project-key`) and click **Create**
4. **Copy the key immediately** — it is only shown once
5. This is your `QDRANT_API_KEY`

> If you lose the key, you'll need to delete it and create a new one.

---

## Step 5: Add to Your `.env` File

Open your `.env` file in the project and fill in:

```env
QDRANT_URL=https://xxxx-xxxx-xxxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here
```

---

## Quick Summary

| What you need | Where to find it |
|---|---|
| `QDRANT_URL` | Cluster Overview → Endpoint field |
| `QDRANT_API_KEY` | Cluster Overview → API Keys tab → Create API Key |
