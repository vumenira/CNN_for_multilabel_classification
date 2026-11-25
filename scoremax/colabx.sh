#!/bin/bash
acct="$1"; shift
PY="$HOME/.local/share/uv/tools/google-colab-cli/bin/python"
lock=/tmp/colabx.lock
while ! mkdir "$lock" 2>/dev/null; do sleep 0.3; done
"$PY" -c "import keyring,sys; s='google-colab-cli'; keyring.set_password(s,'oauth_token', keyring.get_password(s,'oauth_token_'+sys.argv[1]))" "$acct"
( sleep 8; rmdir "$lock" 2>/dev/null ) &
exec "$HOME/.local/bin/colab" --config "$HOME/.config/colab-cli/sessions_$acct.json" "$@"
