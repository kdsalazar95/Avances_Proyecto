#!/usr/bin/env bash
# Lanza Codex: el del PATH si existe; si no, el que trae la extensión de VS Code (la versión más nueva).
# Lo usan las corridas de rl/config/ que corren los Skills 01–09 con Codex (todas salvo corrida_reduccion_claude.json y corrida_reduccion_sin_llm.json).
exe=$(command -v codex 2>/dev/null)
if [ -z "$exe" ]; then
  exe=$(ls -d "$HOME"/.vscode/extensions/openai.chatgpt-*/bin/linux-x86_64/codex 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$exe" ]; then
  echo "No se encontró codex (ni en el PATH ni en la extensión de VS Code)" >&2
  exit 127
fi
exec "$exe" "$@"
