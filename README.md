# MicroMazeMaster

## Quickstart

```bash
poetry install --with dev

```

## Starting the app

CLI

```bash
poetry run micromazemaster
```

GUI

```bash
poetry run micromazemaster_gui
```

## Set the TCL_LIBRARY environment variable on Windows

```
Open System Properties by right-clicking This PC → Properties → Advanced system settings.
Click Environment Variables.
Under System Variables, click New.
    Variable name: TCL_LIBRARY
    Variable value: Path to the Tcl directory (e.g., C:/Users/****/AppData/Local/Programs/Python/Python312/tcl/tcl8.6)
```

## Setup python with tk on MacOS

```bash
brew install python-tk@3.12
```
