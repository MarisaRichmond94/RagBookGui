import { FormEvent, useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import OutlinedInput from "@mui/material/OutlinedInput";
import Paper from "@mui/material/Paper";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

type SourceItem = {
  id: string;
  rank: number;
  source: string;
  metadata: Record<string, unknown>;
  distance: number | null;
  text: string;
  citation: string;
  snippet: string;
  score: number;
  rationale: string;
};

type AskResponse = {
  answer: string;
  sources: SourceItem[];
};

type FilterOptionsResponse = {
  books: string[];
  povs: string[];
};

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(0);
  const [allowedBooks, setAllowedBooks] = useState<string[]>([]);
  const [allowedPovs, setAllowedPovs] = useState<string[]>([]);
  const [selectedBooks, setSelectedBooks] = useState<string[]>([]);
  const [selectedPov, setSelectedPov] = useState("");
  const [rerankSources, setRerankSources] = useState(true);
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

  useEffect(() => {
    async function loadFilterOptions() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/filter-options`);
        if (!response.ok) {
          return;
        }
        const data = (await response.json()) as FilterOptionsResponse;
        setAllowedBooks(data.books ?? []);
        setAllowedPovs(data.povs ?? []);
      } catch {
        setAllowedBooks([]);
        setAllowedPovs([]);
      }
    }

    void loadFilterOptions();
  }, []);

  function onBookSelectChange(event: SelectChangeEvent<string[]>) {
    const value = event.target.value;
    setSelectedBooks(typeof value === "string" ? value.split(",") : value);
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
      const payload: {
        question: string;
        books?: string[];
        pov?: string;
        rerank_sources: boolean;
      } = { question: trimmed, rerank_sources: rerankSources };
      if (selectedBooks.length > 0) {
        payload.books = selectedBooks;
      }
      if (selectedPov) {
        payload.pov = selectedPov;
      }

      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
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
    <Box
      sx={{
        minHeight: "100vh",
        py: { xs: 3, md: 5 },
        px: 2,
        background:
          "radial-gradient(circle at 20% 0%, rgba(76, 201, 240, 0.22), transparent 42%), radial-gradient(circle at 80% 20%, rgba(247, 37, 133, 0.2), transparent 36%), #080b16"
      }}
    >
      <Box sx={{ maxWidth: 980, mx: "auto" }}>
        <Paper elevation={8} sx={{ p: { xs: 2, md: 3 } }}>
          <Stack spacing={2.5}>
            <Typography variant="h4" component="h1" fontWeight={700}>
              RAG Book AI
            </Typography>

            <Box component="form" onSubmit={onSubmit}>
              <Stack spacing={2}>
                <TextField
                  label="Question"
                  multiline
                  minRows={4}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a question about your ragbooks collection..."
                />

                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                  <FormControl fullWidth>
                    <InputLabel id="books-select-label">Book(s)</InputLabel>
                    <Select<string[]>
                      labelId="books-select-label"
                      multiple
                      value={selectedBooks}
                      onChange={onBookSelectChange}
                      input={<OutlinedInput label="Book(s)" />}
                      renderValue={(selected) => (
                        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                          {selected.map((book) => (
                            <Chip key={book} label={book} size="small" />
                          ))}
                        </Stack>
                      )}
                    >
                      {allowedBooks.length === 0 ? (
                        <MenuItem disabled>No books configured</MenuItem>
                      ) : (
                        allowedBooks.map((book) => (
                          <MenuItem key={book} value={book}>
                            {book}
                          </MenuItem>
                        ))
                      )}
                    </Select>
                  </FormControl>

                  <FormControl fullWidth>
                    <InputLabel id="pov-select-label">POV</InputLabel>
                    <Select
                      labelId="pov-select-label"
                      value={selectedPov}
                      label="POV"
                      onChange={(event) => setSelectedPov(event.target.value)}
                    >
                      <MenuItem value="">All</MenuItem>
                      {allowedPovs.map((pov) => (
                        <MenuItem key={pov} value={pov}>
                          {pov}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Stack>

                <FormControlLabel
                  control={
                    <Switch
                      checked={rerankSources}
                      onChange={(_, checked) => setRerankSources(checked)}
                    />
                  }
                  label="Rerank sources"
                />

                <Button type="submit" variant="contained" size="large" disabled={isLoading}>
                  {isLoading ? "Asking..." : "Ask"}
                </Button>
              </Stack>
            </Box>

            {error ? <Alert severity="error">{error}</Alert> : null}

            {answer ? (
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Answer
                </Typography>
                <Typography variant="body1" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.65 }}>
                  {answer}
                </Typography>
              </Paper>
            ) : null}

            {sources.length > 0 ? (
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Sources
                </Typography>

                <Tabs
                  value={selectedSourceIndex}
                  onChange={(_, value: number) => setSelectedSourceIndex(value)}
                  variant="scrollable"
                  scrollButtons="auto"
                  sx={{ mb: 2, borderBottom: 1, borderColor: "divider" }}
                >
                  {sources.map((item, idx) => (
                    <Tab key={`${item.rank}-${idx}`} label={sourceLabel(item)} />
                  ))}
                </Tabs>

                {selectedSource ? (
                  <Stack spacing={1.5}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Score: {selectedSource.score.toFixed(2)}
                      {selectedSource.distance !== null
                        ? `  |  Distance: ${selectedSource.distance.toFixed(4)}`
                        : ""}
                    </Typography>

                    <Paper variant="outlined" sx={{ p: 1.5 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Citation
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1.5 }}>
                        {selectedSource.citation}
                      </Typography>

                      <Typography variant="subtitle2" gutterBottom>
                        Metadata
                      </Typography>
                      <Stack direction={{ xs: "column", sm: "row" }} spacing={3}>
                        <Typography variant="body2">
                          <strong>Date:</strong>{" "}
                          {metadataValue(metadataField(selectedSource.metadata, "date") ?? "N/A")}
                        </Typography>
                        <Typography variant="body2">
                          <strong>POV:</strong>{" "}
                          {metadataValue(metadataField(selectedSource.metadata, "pov") ?? "N/A")}
                        </Typography>
                      </Stack>
                    </Paper>

                    <Paper variant="outlined" sx={{ p: 1.5, maxHeight: 200, overflowY: "auto" }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Snippet
                      </Typography>
                      <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6, mb: 1 }}>
                        {selectedSource.snippet}
                      </Typography>

                      <Typography variant="subtitle2" gutterBottom>
                        Sourced Text
                      </Typography>
                      <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                        {selectedSource.text}
                      </Typography>
                    </Paper>
                  </Stack>
                ) : null}
              </Paper>
            ) : null}
          </Stack>
        </Paper>
      </Box>
    </Box>
  );
}
