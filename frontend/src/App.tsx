import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service: string;
};

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [message, setMessage] = useState("Loading...");

  useEffect(() => {
    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data: HealthResponse = await response.json();
        setMessage(`${data.service}: ${data.status}`);
      } catch {
        setMessage("Could not reach backend.");
      }
    }

    void checkBackend();
  }, []);

  return (
    <main className="container">
      <h1>Vite + React + FastAPI Monorepo</h1>
      <p>Backend status: {message}</p>
    </main>
  );
}

