import React from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./app/globals.css";
import Home from "./app/page";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Home />
  </React.StrictMode>,
);
