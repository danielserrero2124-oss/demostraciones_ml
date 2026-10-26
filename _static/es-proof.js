// Traduce la etiqueta "Proof." de sphinx-proof al español.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("div.proof#proof > p:first-child").forEach(function (p) {
    var walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
    var node = walker.nextNode();
    if (node && /^\s*Proof\./.test(node.nodeValue)) {
      node.nodeValue = node.nodeValue.replace(/^(\s*)Proof\./, "$1Demostración.");
    }
  });
});
