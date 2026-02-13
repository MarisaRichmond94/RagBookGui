import { FormEvent, useState } from "react";

type SourceItem = {
  source: string;
  metadata: Record<string, unknown>;
  distance: number | null;
};

type AskResponse = {
  answer: string;
  sources: SourceItem[];
};

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setAnswer("");
    setSources([]);

    const trimmed = question.trim();
    if (!trimmed) {
      setError("Please enter a question.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed })
      });

      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? "Request failed.");
      }

      const data = (await response.json()) as AskResponse;
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Request failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>RAG Books QA</h1>

      <form className="ask-form" onSubmit={onSubmit}>
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={5}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your ragbooks collection..."
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Asking..." : "Ask"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {answer ? (
        <section className="panel">
          <h2>Answer</h2>
          <p>{answer}</p>
        </section>
      ) : null}

      {sources.length > 0 ? (
        <section className="panel">
          <h2>Sources</h2>
          <ul className="sources">
            {sources.map((item, idx) => (
              <li key={`${item.source}-${idx}`}>
                <strong>{item.source}</strong>
                {item.distance !== null ? (
                  <span className="distance">distance: {item.distance.toFixed(4)}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}
