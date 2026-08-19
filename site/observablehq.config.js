// Halfcell — Observable Framework config
// Static build of the GB BESS market analysis previously served via Streamlit.

const title = "Halfcell";

// Absolute, because unfurlers do not resolve relative og:image paths.
// Regenerate the card with scripts/make_og_image.py after a data refresh.
const siteUrl = "https://halfcell.pages.dev";
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
<meta property="og:url" content="${siteUrl}/">
<meta property="og:image" content="${siteUrl}/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Halfcell — cumulative modelled revenue for three forecasting strategies on a GB grid-scale battery">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="${siteUrl}/og-image.png">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>%F0%9F%94%8B</text></svg>">
`,

  // Custom stylesheet rather than a stock theme — see src/style.css
  style: "style.css",

  // Framework loads Source Serif 4 from Google Fonts by default. The palette in
  // style.css uses local system serifs instead, so that request fetched a font
  // nothing rendered with — a third-party round trip on every page load for
  // nothing. Empty array removes it.
  globalStylesheets: [],
  toc: false,
  pager: false,
  footer: () =>
    `Data through ${new Date().getFullYear()} · ` +
    `<a href="https://github.com/finbarrhodes/Halfcell">GitHub</a> · ` +
    `<a href="https://www.linkedin.com/in/finbar-rhodes-637650210/">LinkedIn</a>`,
};
