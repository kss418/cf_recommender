import React, { useMemo, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${Math.round(Number(value) * 100)}%`;
}

function getTagBarWidth(tagSkill, tagSkills) {
  if (!tagSkills.length) {
    return 0;
  }

  const ratings = tagSkills.map((tag) => tag.skill_rating);
  const minRating = Math.min(...ratings);
  const maxRating = Math.max(...ratings);
  if (maxRating === minRating) {
    return 100;
  }

  return 100 - ((tagSkill.skill_rating - minRating) / (maxRating - minRating)) * 72;
}

function App() {
  const [handle, setHandle] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [selectedTag, setSelectedTag] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const tagSkills = useMemo(
    () => analysis?.tag_skills ?? [],
    [analysis],
  );

  const selectedProblems = useMemo(() => {
    if (!selectedTag || !analysis?.problems_by_tag) {
      return [];
    }

    return analysis.problems_by_tag[selectedTag] ?? [];
  }, [analysis, selectedTag]);

  async function analyzeHandle(event) {
    event.preventDefault();
    const trimmedHandle = handle.trim();
    if (!trimmedHandle) {
      return;
    }

    setLoading(true);
    setError("");
    setAnalysis(null);
    setSelectedTag(null);

    try {
      const response = await fetch(`${API_BASE_URL}/recommend/${trimmedHandle}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Request failed");
      }

      setAnalysis(payload);
      setSelectedTag(payload.tag_skills?.[0]?.tag ?? null);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <p className="eyebrow">Codeforces Recommender</p>
          <h1>Weakness Map</h1>
        </div>
        <form className="search-form" onSubmit={analyzeHandle}>
          <label htmlFor="handle">Handle</label>
          <div className="search-row">
            <input
              id="handle"
              value={handle}
              onChange={(event) => setHandle(event.target.value)}
              placeholder="TRErnD"
              spellCheck="false"
            />
            <button type="submit" disabled={loading}>
              {loading ? "Analyzing" : "Analyze"}
            </button>
          </div>
        </form>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {analysis ? (
        <section className="workspace">
          <aside className="summary-panel">
            <div className="metric">
              <span>Handle</span>
              <strong>{analysis.user_id}</strong>
            </div>
            <div className="metric">
              <span>Rating</span>
              <strong>{analysis.user_rating ?? "-"}</strong>
            </div>
            <div className="metric">
              <span>Submissions</span>
              <strong>{analysis.user_data?.submission_count ?? "-"}</strong>
            </div>
            <div className="metric">
              <span>Candidates</span>
              <strong>{analysis.candidate_count}</strong>
            </div>
          </aside>

          <section className="chart-panel">
            <div className="panel-header">
              <h2>Tag Skill</h2>
              <span>{tagSkills.length} tags</span>
            </div>
            <div className="tag-list">
              {tagSkills.map((tagSkill) => (
                <button
                  className={
                    tagSkill.tag === selectedTag
                      ? "tag-row selected"
                      : "tag-row"
                  }
                  key={tagSkill.tag}
                  onClick={() => setSelectedTag(tagSkill.tag)}
                  type="button"
                >
                  <span className="tag-name">{tagSkill.tag}</span>
                  <span className="bar-track">
                    <span
                      className="bar-fill"
                      style={{ width: `${getTagBarWidth(tagSkill, tagSkills)}%` }}
                    />
                  </span>
                  <span className="tag-rating">
                    {formatNumber(tagSkill.skill_rating, 0)}
                  </span>
                  <span className="tag-meta">
                    {tagSkill.exposure} · {formatPercent(tagSkill.confidence)}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section className="problem-panel">
            <div className="panel-header">
              <h2>{selectedTag ?? "Problems"}</h2>
              <span>Top 5</span>
            </div>
            <div className="problem-list">
              {selectedProblems.map((problem) => (
                <a
                  className="problem-item"
                  href={problem.url}
                  key={`${problem.contest_id}${problem.index}`}
                  rel="noreferrer"
                  target="_blank"
                >
                  <div>
                    <div className="problem-title">
                      {problem.contest_id}
                      {problem.index} · {problem.name}
                    </div>
                    <div className="problem-tags">
                      {problem.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  </div>
                  <div className="problem-stats">
                    <strong>{problem.rating}</strong>
                    <span>p {formatPercent(problem.solve_probability)}</span>
                    <span>score {formatNumber(problem.score, 3)}</span>
                  </div>
                </a>
              ))}
              {!selectedProblems.length && (
                <div className="empty-state">No candidates</div>
              )}
            </div>
          </section>
        </section>
      ) : (
        <section className="empty-dashboard">
          <div className="empty-metric" />
          <div className="empty-chart" />
          <div className="empty-list" />
        </section>
      )}
    </main>
  );
}

export default App;
