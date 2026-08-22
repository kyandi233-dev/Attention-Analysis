# clipboard-vision-mcp

> 🇬🇧 **[English version → README.md](README.md)**

> Ajoute la vision aux modèles texte-only dans Opencode (**DeepSeek V4**, **GLM 5.1**) — **lis directement l'image dans ton presse-papiers**, sans sauvegarder de fichier à la main.

**Testé sur Windows 11 + Opencode + DeepSeek V4 Pro.** Support clipboard multi-OS (Windows / macOS / Linux X11 / Linux Wayland).

Forké depuis [itcomgroup/vision-mcp-server](https://github.com/itcomgroup/vision-mcp-server) — réécrit autour d'outils clipboard-first, durcissement sécurité, extraction clipboard cross-platform, et installation via prompt IA.

---

## Le problème

Les modèles texte-only rapides et peu coûteux comme **DeepSeek V4** et **GLM 5.1** sont excellents pour le code mais ne savent pas lire d'images. À chaque capture d'écran collée, le modèle te demande de la sauvegarder sur disque et de lui donner le chemin.

## La solution

Ce serveur MCP expose des outils `*_from_clipboard`. Quand le LLM veut voir ta capture, il appelle `analyze_clipboard` — le serveur lit l'image du presse-papiers, l'envoie à un vrai modèle de vision (**Groq + Llama-4 Scout, gratuit**), et renvoie une description texte que le modèle peut exploiter.

Résultat : **copier → demander → terminé.** Aucun fichier à manipuler.

---

## 🤖 Installation en un prompt via IA (recommandé)

Au lieu de faire les étapes manuelles ci-dessous, colle un de ces prompts dans n'importe quel assistant de code (DeepSeek, GLM, Claude, GPT, ...) et il fait tout de A à Z — clone, venv, dépendances, config MCP, raccourcis :

- 🇫🇷 **[docs/INSTALL_PROMPT_FR.md](docs/INSTALL_PROMPT_FR.md)**
- 🇬🇧 **[docs/INSTALL_PROMPT_EN.md](docs/INSTALL_PROMPT_EN.md)**

Tu préfères installer toi-même ? Continue à lire.

---

## Fonctionnalités

- 🖼️ **Clipboard-first** — `analyze_clipboard`, `extract_text_from_clipboard`, `diagnose_error_from_clipboard`, `describe_ui_from_clipboard`, `code_from_clipboard`.
- 📁 **Fallback fichier** — mêmes outils disponibles pour les images déjà sur disque.
- 🆓 **Vision gratuite** — Groq free tier avec Llama-4 Scout (17B, multimodal).
- 🖥️ **Multi-OS** — Windows, macOS, Linux (X11 + Wayland).
- 🔒 **Sécurisé** — validation extension/taille/magic-bytes, suppression auto des fichiers clipboard temporaires.
- 🔌 **Standard MCP** — fonctionne avec Opencode, Claude Code, Cursor, Cline, Continue, ou tout client MCP.

---

## Prérequis

- **Python 3.10+**
- **Clé API Groq** (gratuite, 30 secondes) : https://console.groq.com/keys
- Un client MCP (Opencode, Claude Code, Cursor, Cline, Continue, ...)

### Dépendances Python (installées automatiquement via `pip install -e .`)

| Paquet | Rôle |
|---|---|
| `mcp>=1.0.0` | Serveur de protocole MCP |
| `groq>=0.11.0` | Client API Groq (vision Llama-4 Scout) |
| `aiofiles>=23.0.0` | I/O fichier asynchrone |
| `Pillow>=10.0.0` | Extraction clipboard (Windows/macOS), encodage PNG |

### Dépendances clipboard par OS

| OS | Commande | Pourquoi |
|---|---|---|
| **Windows** | *rien à installer* | Pillow + pywin32 gèrent le clipboard nativement. |
| **macOS** | `brew install pngpaste` *(fallback optionnel)* | Pillow suffit la plupart du temps. |
| **Linux — Wayland** | `sudo apt install wl-clipboard` | Fournit `wl-paste`. |
| **Linux — X11** | `sudo apt install xclip` | Ou équivalent pour ta distro. |

---

## Démarrage rapide

### 1. Récupère une clé Groq gratuite
https://console.groq.com/keys

### 2. Installation

```bash
git clone https://github.com/Capetlevrai/clipboard-vision-mcp.git
cd clipboard-vision-mcp
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS/Linux :
source .venv/bin/activate
pip install -e .
```

### 3. Smoke test (sans Groq)

Copie une capture d'écran, puis :

```bash
python examples/smoke_test.py
```

Attendu : `OK: clipboard image saved to <chemin>`.

### 4. Branche-le à ton client MCP

Voir [docs/OPENCODE.md](docs/OPENCODE.md) pour Opencode (testé Windows) ou [docs/CLIENTS.md](docs/CLIENTS.md) pour Claude Code / Cursor / Cline / Continue.

**Opencode** (`%APPDATA%\opencode\opencode.json` sous Windows, `~/.config/opencode/opencode.json` sous Linux/macOS) :

```json
{
  "mcp": {
    "clipboard-vision": {
      "type": "local",
      "command": [
        "C:\\chemin\\vers\\clipboard-vision-mcp\\.venv\\Scripts\\python.exe",
        "-m",
        "clipboard_vision_mcp"
      ],
      "enabled": true,
      "environment": {
        "GROQ_API_KEY": "gsk_ta_cle_ici"
      }
    }
  }
}
```

> 💡 **Utilise le chemin absolu du Python du venv.** Ça garantit que le MCP démarre avec les bonnes dépendances, peu importe le shell, le cwd ou le venv actif.

### 5. ⚠️ Raccourcis Opencode pour coller des images (Alt+V)

Deux choses importantes sur opencode (faciles à se tromper) :

- Les raccourcis vivent dans **`tui.json`** (à côté d'`opencode.json`), **PAS** dans `opencode.json`. Ce dernier rejette les clés inconnues, donc y mettre un bloc `keybinds` fait **planter le démarrage** d'opencode (`ConfigInvalidError`).
- Il n'existe **pas** d'action `input_paste_image`. Le collage (texte **et** image) est l'action unique `input_paste` — le TUI d'opencode lit directement l'image dans le presse-papier OS. Elle est liée à `Ctrl+V` par défaut ; pour coller aussi avec `Alt+V`, lie `input_paste` aux deux.

Crée/fusionne `tui.json` (ex. `~/.config/opencode/tui.json` sous Linux/macOS, `C:\Users\<toi>\.config\opencode\tui.json` sous Windows) :

```json
{
  "$schema": "https://opencode.ai/tui.json",
  "keybinds": {
    "input_paste": [
      { "key": "ctrl+v", "preventDefault": false },
      { "key": "alt+v", "preventDefault": false }
    ]
  }
}
```

Redémarre Opencode. Copie ensuite une capture et appuie sur `Alt+V` (ou `Ctrl+V`) dans la zone de saisie — l'image s'attache.

> ℹ️ Optionnel : les outils `*_from_clipboard` lisent eux-mêmes le presse-papier OS, tu peux donc zapper le collage — copie une capture et demande au modèle « analyse mon presse-papier ».

### 6. Est-ce que ça démarre automatiquement après un redémarrage Windows ?

**Oui.** Opencode relit `opencode.json` à chaque lancement et démarre automatiquement tout serveur MCP avec `"type": "local"` et `"enabled": true`. Comme la commande utilise le **chemin absolu du Python du venv**, peu importe depuis quel shell ou dossier Opencode est lancé.

Redémarre le PC → ouvre Opencode → les outils `clipboard-vision` sont listés. Zéro manip manuelle.

---

## Utilisation

```
Toi : (copie une capture, puis tape)
      "Regarde ce que je viens de copier et dis-moi quelle est cette erreur."

LLM (DeepSeek, GLM, Claude, ...): [appelle diagnose_error_from_clipboard]
      → "L'erreur dit `ECONNREFUSED 127.0.0.1:5432`. Postgres n'écoute
         pas sur le port 5432. Démarre-le avec : ..."
```

Le modèle texte-only ne voit jamais les pixels — il lit la description renvoyée par Llama-4 Scout et raisonne dessus.

### Référence des outils

| Outil | Entrée | Quand l'utiliser |
|---|---|---|
| `analyze_clipboard` | `prompt` optionnel | Description générique, questions sur l'image. |
| `extract_text_from_clipboard` | — | OCR pur. |
| `describe_ui_from_clipboard` | — | Revue UI/UX, inventaire de composants. |
| `diagnose_error_from_clipboard` | — | Capture d'erreur → cause + correctif. |
| `code_from_clipboard` | — | Extraire du code d'une capture. |
| `analyze_image` | `image_path`, `prompt` optionnel | Image déjà sur disque. |
| `extract_text`, `describe_ui`, `diagnose_error`, `understand_diagram`, `analyze_chart`, `code_from_screenshot` | `image_path` | Versions fichier des outils ci-dessus. |

---

## Sécurité

Le serveur tourne en **processus local stdio** — aucun port réseau ouvert, il ne parle qu'au client MCP (via stdin/stdout) et à l'API Groq (via HTTPS).

Protections en place :

- **Allow-list d'extensions.** `analyze_image` et les autres outils fichier n'acceptent que `.png .jpg .jpeg .gif .webp .bmp`. Ça empêche un LLM prompt-injecté de demander au serveur de lire un fichier quelconque (`~/.ssh/id_rsa`, `.env`, ...) et de l'exfiltrer en base64 vers Groq.
- **Vérification magic-bytes.** Le contenu du fichier est validé contre des en-têtes d'image connus avant upload.
- **Limite de taille.** 20 Mo max par image.
- **Suppression automatique des fichiers clipboard temporaires** après chaque analyse. Les captures peuvent contenir des secrets (tokens, chats, identifiants) — le serveur les écrit dans `$TMPDIR/clipboard_vision_mcp/` puis les supprime une fois l'analyse terminée.
- **Aucune télémétrie.**

### Ce que le projet ne peut PAS te garantir

- **Ta clé API vit en clair dans ta config client MCP.** C'est comme ça que les clients MCP fonctionnent aujourd'hui. Garde ce fichier non-lisible par les autres et ne le commite jamais. Si tu exposes une clé par accident (chat, capture, git push), **rotate-la** sur https://console.groq.com/keys.
- **Groq reçoit les images analysées.** Consulte leur [politique de confidentialité](https://groq.com/privacy-policy/) avant d'envoyer du sensible.
- **Tout outil MCP est exécuté à la demande du LLM.** Si tu connectes un modèle prompt-injecté à ce serveur et lui fais lire de l'input non fiable, le modèle choisit ce qu'il analyse. L'allow-list ci-dessus limite les dégâts mais ne peut pas les éliminer.

### Signaler une faille

Ouvre un [security advisory privé](https://github.com/Capetlevrai/clipboard-vision-mcp/security/advisories/new) plutôt qu'une issue publique.

---

## Fonctionnement

```
┌──────────────┐   MCP   ┌─────────────────┐   HTTPS   ┌─────────────────┐
│  Opencode    │ ──────▶ │  clipboard-     │ ────────▶ │  Groq API       │
│  (DeepSeek)  │         │  vision-mcp     │           │  Llama-4 Scout  │
└──────────────┘         └─────────────────┘           └─────────────────┘
                              │
                              ▼
                    lit le presse-papiers système
                    (PIL / wl-paste / xclip)
                    → valide → base64 → envoie → supprime
```

---

## Dépannage

- **« Clipboard does not contain an image. »** — Copie une vraie image, pas un icône de fichier ou du texte. Sous Linux, teste `wl-paste --type image/png` ou `xclip -selection clipboard -t image/png -o | file -` hors du MCP.
- **« GROQ_API_KEY is not set. »** — Vérifie le bloc `environment` dans la config client, puis **redémarre complètement** le client.
- **Les outils n'apparaissent pas dans Opencode.** — Regarde les logs MCP d'Opencode. Lance `python -m clipboard_vision_mcp` à la main — il doit démarrer et rester silencieux sur stdin.
- **« Refusing to read '<ext>' — only image files are allowed. »** — Tu (ou le LLM) a passé un chemin non-image. C'est le garde-fou sécurité qui fait son boulot.

---

## Crédits

- Forké depuis [itcomgroup/vision-mcp-server](https://github.com/itcomgroup/vision-mcp-server) — intégration Groq + Llama-4 Scout originale.
- Modèle de vision : [Llama-4 Scout 17B](https://groq.com/) servi par Groq.

## Licence

MIT — voir [LICENSE](LICENSE).
