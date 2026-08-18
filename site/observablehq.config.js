// Halfcell — Observable Framework config
// Static build of the GB BESS market analysis previously served via Streamlit.

const title = "Halfcell";
const description =
  "Analysis of GB battery storage markets — frequency response clearing prices, " +
  "wholesale arbitrage, and a day-ahead forecasting and MPC dispatch model.";

export default {
  title,
  root: "src",

  pages: [
    {name: "Market Overview", path: "/dashboard"},
    {name: "Forecasting & Dispatch", path: "/backtester"},
    {name: "Methodology & Data", path: "/methodology"},
  ],

  // Open Graph tags so a pasted link renders as a proper preview card —
  // the main reason this is worth controlling ourselves rather than
  // inheriting a host's generic default.
  head: `
<meta name="description" content="${description}">
<meta property="og:title" content="${title} — GB battery storage market analysis">
<meta property="og:description" content="${description}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>%F0%9F%94%8B</text></svg>">
`,

  theme: "air",
  toc: false,
  pager: false,
  footer: () =>
    `Data through ${new Date().getFullYear()} · ` +
    `<a href="https://github.com/finbarrhodes/Halfcell">GitHub</a> · ` +
    `<a href="https://www.linkedin.com/in/finbar-rhodes-637650210/">LinkedIn</a>`,
};
