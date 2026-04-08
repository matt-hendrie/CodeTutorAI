// CodeTutorAI - Main JavaScript

document.addEventListener("DOMContentLoaded", () => {
  console.log("CodeTutorAI loaded");

  // HTMX configuration overrides (supplements meta tag config)
  if (typeof htmx !== "undefined") {
    // Log HTMX events for debugging
    htmx.on("htmx:afterRequest", (event) => {
      console.log(
        `HTMX request completed: ${event.detail.pathInfo.requestPath}`,
      );
    });

    htmx.on("htmx:responseError", (event) => {
      console.error(
        `HTMX response error: ${event.detail.xhr.status}`,
        event.detail,
      );
    });

    htmx.on("htmx:sendError", (event) => {
      console.error("HTMX send error — check network connection", event.detail);
    });

    console.log("HTMX initialized", htmx.config);
  }
});
