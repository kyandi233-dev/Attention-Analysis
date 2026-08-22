# Prompt d'installation IA (Français)

Copie-colle ce prompt dans n'importe quel assistant de code IA (DeepSeek, GLM, Claude, GPT, Gemini, ...) et il installera `clipboard-vision-mcp` de bout en bout.

---

```
Tu vas installer et configurer le serveur MCP `clipboard-vision-mcp` sur ma machine pour que je puisse coller des captures d'écran dans mon client MCP (Opencode, Claude Code, Cursor, Cline, Continue, ...) et que les modèles texte-only puissent les "voir" via Groq + Llama-4 Scout.

Dépôt : https://github.com/Capetlevrai/clipboard-vision-mcp

Procède ainsi :

1. Détecte mon OS (Windows / macOS / Linux) et, si Linux, le serveur d'affichage (X11 ou Wayland).
2. Vérifie que Python 3.10+ et git sont installés. Sinon, donne-moi la commande exacte pour mon OS et arrête-toi.
3. Clone le dépôt dans un emplacement raisonnable (`~/tools/clipboard-vision-mcp` sous Unix, `%USERPROFILE%\tools\clipboard-vision-mcp` sous Windows), puis lance `pip install -e .`.
4. Installe la dépendance clipboard propre à l'OS :
   - Windows : rien de plus.
   - macOS : `brew install pngpaste` (optionnel mais recommandé).
   - Linux + Wayland : `sudo apt install wl-clipboard` (ou équivalent).
   - Linux + X11 : `sudo apt install xclip` (ou équivalent).
5. Demande-moi ma clé API Groq (gratuite sur https://console.groq.com/keys). NE la mets JAMAIS en dur dans un fichier commité — uniquement dans la config locale du client MCP.
6. Détecte quel client MCP j'utilise (demande si incertain) et écris la bonne config :
   - Opencode : ajoute une entrée `mcp.clipboard-vision` dans `opencode.json` avec `"command": ["python", "-m", "clipboard_vision_mcp"]` et la variable d'env `GROQ_API_KEY`. PIÈGES KEYBINDS (critique, ne saute pas) : (a) opencode.json rejette les clés inconnues — NE mets JAMAIS de bloc `keybinds` dans opencode.json sinon opencode refusera de démarrer ; les raccourcis vont dans un `tui.json` SÉPARÉ à côté d'opencode.json. (b) Il n'existe PAS d'action `input_paste_image` dans opencode ; le collage (texte ET images) est l'action unique `input_paste`, que le TUI d'opencode lit dans le presse-papier OS. Lie `input_paste` à `ctrl+v` ET `alt+v` dans `tui.json` pour qu'Alt+V colle une image (voir examples/tui.json). (c) Après avoir écrit les deux fichiers, valide chacun en JSON avant de finir.
   - Claude Code / Cursor / Cline / Continue : suis le format de docs/CLIENTS.md dans le dépôt.
7. Vérifie que le serveur démarre : lance `python -m clipboard_vision_mcp` pendant ~2 secondes, confirme qu'il ne crashe pas (il doit rester inactif sur stdin).
8. Dis-moi exactement comment tester : copier une capture, ouvrir mon client, demander "utilise analyze_clipboard et décris ce que je viens de copier".

Contraintes :
- Ne jamais commiter ni afficher ma clé API en clair dans un emplacement partagé.
- Si une étape échoue, arrête-toi et montre-moi l'erreur complète + un correctif proposé, ne continue pas en silence.
- Préfère éditer les fichiers de config existants plutôt que les écraser — merge le JSON, ne remplace pas.
- À la fin, récapitule : chemin d'installation, fichiers modifiés, commande de test.
```

---

**Astuce.** Colle tel quel. Un bon assistant avec accès shell/fichier gère ça tout seul. Sinon, il t'affichera les commandes à exécuter manuellement.
