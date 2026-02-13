import { FormEvent, useState } from "react";

type SourceItem = {
  rank: number;
  source: string;
  metadata: Record<string, unknown>;
  distance: number | null;
  text: string;
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
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(0);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const selectedSource = sources[selectedSourceIndex] ?? null;

  function metadataValue(value: unknown): string {
    if (typeof value === "string") {
      return value;
    }
    return JSON.stringify(value);
  }

  function metadataField(metadata: Record<string, unknown>, name: string): unknown {
    const exact = metadata[name];
    if (exact !== undefined) {
      return exact;
    }

    const match = Object.entries(metadata).find(([key]) => key.toLowerCase() === name.toLowerCase());
    return match?.[1];
  }

  function sourceLabel(item: SourceItem): string {
    const metadata = item.metadata ?? {};
    const book = String(
      metadata.book ??
        metadata.title ??
        metadata.book_title ??
        metadata.source ??
        item.source ??
        `Document ${item.rank}`
    );
    const chapterValue =
      metadata.chapter ?? metadata.chapter_number ?? metadata.chapter_num ?? metadata.section;
    const chapter = chapterValue === undefined || chapterValue === null ? "?" : String(chapterValue);
    return `${book} - Chapter ${chapter}`;
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setAnswer("");
    setSources([]);
    setSelectedSourceIndex(0);

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
      setSelectedSourceIndex(0);
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
          <div className="source-tabs" role="tablist" aria-label="Source list">
            {sources.map((item, idx) => (
              <button
                key={`${item.rank}-${idx}`}
                type="button"
                className={`source-tab ${selectedSourceIndex === idx ? "active" : ""}`}
                onClick={() => setSelectedSourceIndex(idx)}
              >
                {sourceLabel(item)}
              </button>
            ))}
          </div>

          {selectedSource ? (
            <div className="source-detail">
              <p className="source-meta-line">
                <strong>Selected:</strong> {sourceLabel(selectedSource)}
                {selectedSource.distance !== null ? (
                  <span className="distance">
                    distance: {selectedSource.distance.toFixed(4)}
                  </span>
                ) : null}
              </p>

              <div className="metadata-block">
                <h3>Metadata</h3>
                <dl className="metadata-list">
                  <div className="metadata-item">
                    <dt>date</dt>
                    <dd>
                      {metadataValue(metadataField(selectedSource.metadata, "date") ?? "N/A")}
                    </dd>
                  </div>
                  <div className="metadata-item">
                    <dt>pov</dt>
                    <dd>
                      {metadataValue(metadataField(selectedSource.metadata, "pov") ?? "N/A")}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="text-block">
                <h3>Sourced Text</h3>
                <p>{selectedSource.text}</p>
              </div>
            </div>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
