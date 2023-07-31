from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="TAKEOFF",
    settings_files=["app/settings.yaml", "app/.secrets.yaml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
