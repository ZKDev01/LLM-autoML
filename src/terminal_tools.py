# Formato de impresión para el CLI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Funciones de impresión con formato
def ok(msg): print(f"  {GREEN}✔{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{CYAN}{msg}{RESET}")
