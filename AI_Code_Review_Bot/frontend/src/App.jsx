import { useState } from "react";

import { reviewCode, reviewPr, reviewRepo } from "./api";

const emptyResult = null;

export default function App() {
  const [tab, setTab] = useState("code");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(emptyResult);
  const [error, setError] = useState("");
  const [codePayload, setCodePayload] = useState({
    filename: "snippet.py",
    language: "python",
    code: "def add(a, b):\n    return a + b\n",
  });
  const [repoUrl, setRepoUrl] = useState("");
  const [prUrl, setPrUrl] = useState("");

  async function handleSubmit() {
    setLoading(true);
    setError("");

    try {
      const data =
        tab === "code"
          ? await reviewCode(codePayload)
          : tab === "repo"
            ? await reviewRepo({ repo_url: repoUrl })
            : await reviewPr({ pr_url: prUrl });
      setResult(data);
    } catch (submissionError) {
      setError(submissionError.message || "Review request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Multi-Agent LangGraph Platform</p>
          <h1>AI Code Review Bot</h1>
          <p className="lede">
            Review raw code, repositories, or pull requests through a specialist agent pipeline
            for bugs, complexity, security, tests, documentation, scoring, and final synthesis.
          </p>
        </div>
        <div className="hero-card">
          <span>Agents</span>
          <strong>9</strong>
          <small>Analyzer to Reviewer</small>
        </div>
      </header>

      <main className="grid">
        <section className="panel input-panel">
          <div className="tab-row">
            {[
              ["code", "Review Code"],
              ["repo", "Review Repo"],
              ["pr", "Review PR"],
            ].map(([value, label]) => (
              <button
                key={value}
                className={tab === value ? "tab active" : "tab"}
                onClick={() => setTab(value)}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "code" && (
            <div className="form-stack">
              <label>
                Filename
                <input
                  value={codePayload.filename}
                  onChange={(event) =>
                    setCodePayload((current) => ({ ...current, filename: event.target.value }))
                  }
                />
              </label>
              <label>
                Language
                <input
                  value={codePayload.language}
                  onChange={(event) =>
                    setCodePayload((current) => ({ ...current, language: event.target.value }))
                  }
                />
              </label>
              <label>
                Source Code
                <textarea
                  rows={16}
                  value={codePayload.code}
                  onChange={(event) =>
                    setCodePayload((current) => ({ ...current, code: event.target.value }))
                  }
                />
              </label>
            </div>
          )}

          {tab === "repo" && (
            <div className="form-stack">
              <label>
                GitHub Repository URL
                <input
                  placeholder="https://github.com/org/repo"
                  value={repoUrl}
                  onChange={(event) => setRepoUrl(event.target.value)}
                />
              </label>
            </div>
          )}

          {tab === "pr" && (
            <div className="form-stack">
              <label>
                GitHub Pull Request URL
                <input
                  placeholder="https://github.com/org/repo/pull/123"
                  value={prUrl}
                  onChange={(event) => setPrUrl(event.target.value)}
                />
              </label>
            </div>
          )}

          <button className="submit" type="button" onClick={handleSubmit} disabled={loading}>
            {loading ? "Review in progress..." : "Run Multi-Agent Review"}
          </button>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel result-panel">
          <div className="result-header">
            <h2>Structured Report</h2>
            {result ? <span>Score {result.report.code_quality_score}/10</span> : null}
          </div>
          {result ? (
            <pre>{JSON.stringify(result, null, 2)}</pre>
          ) : (
            <p className="placeholder">
              The final reviewer output will appear here as structured JSON.
            </p>
          )}
        </section>
      </main>
    </div>
  );
}
