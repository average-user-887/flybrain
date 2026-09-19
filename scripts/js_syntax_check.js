// Parse-only check for a browser JS file, run under gjs (SpiderMonkey):
//   gjs scripts/js_syntax_check.js web/app.js
const GLib = imports.gi.GLib;
const path = ARGV[0];
const bytes = GLib.file_get_contents(path)[1];
const text = new TextDecoder().decode(bytes);
try {
    new Function(text);
    print(`PARSE OK: ${path}`);
} catch (e) {
    print(`PARSE ERROR in ${path}: ${e.message} (line ${e.lineNumber})`);
    imports.system.exit(1);
}
