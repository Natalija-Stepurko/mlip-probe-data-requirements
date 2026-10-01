// Load the built page in jsdom, run its scripts, exercise the calculator, and exit 1 on a runtime
// error or an interactive element left empty.   Run with `make check-page`.
const {JSDOM, VirtualConsole} = require("jsdom"), fs = require("fs");
const errors = [];
const vc = new VirtualConsole().on("jsdomError", e => errors.push(e.message));
const dom = new JSDOM(fs.readFileSync(process.argv[2], "utf8"), {runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc});
const d = dom.window.document, $ = id => d.getElementById(id), text = id => ($(id) ? $(id).textContent.trim() : "");

const sel = $("tTarget");
sel.value = "crystal_system"; sel.dispatchEvent(new dom.window.Event("change"));
d.querySelector('.try-seg-b[data-emb="uma"]').click();

const checks = {
  "calculator targets": sel.options.length > 0,
  "calculator output": ["vScore", "vShort", "vSpread", "vTotal"].every(id => /\d/.test(text(id))),
  "split explorer": $("sfxChart").children.length > 0 && text("sfxCap") !== "",
  "cross-validation explorer": $("cvxChart").children.length > 0 || text("cvxCap") !== "",   // captioned when not measured
  "score panels": $("figScore").querySelectorAll("figure.panel").length > 0,
  "shortfall panels": $("figGap").querySelectorAll("figure.panel").length > 0,
  "live numbers": [...d.querySelectorAll("[data-live]")].every(e => e.textContent.trim() !== ""),
};
errors.forEach(e => console.log("ERROR", e));
const empty = Object.keys(checks).filter(k => !checks[k]);
empty.forEach(k => console.log("EMPTY", k));
console.log(errors.length || empty.length ? "page check failed" : `page check passed (${Object.keys(checks).length} checks)`);
process.exit(errors.length || empty.length ? 1 : 0);
