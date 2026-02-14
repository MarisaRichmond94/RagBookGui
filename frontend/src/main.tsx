import React from "react";
import ReactDOM from "react-dom/client";
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import App from "./App";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#4cc9f0"
    },
    secondary: {
      main: "#f72585"
    },
    background: {
      default: "#080b16",
      paper: "#10182b"
    }
  },
  shape: {
    borderRadius: 12
  },
  typography: {
    fontFamily: "\"Space Grotesk\", \"Avenir Next\", \"Segoe UI\", sans-serif"
  }
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
