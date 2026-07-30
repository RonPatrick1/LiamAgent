"""GNOME Keyring storage for desktop SSH sudo credentials.

This module is intentionally not registered as an agent tool.  Only the
desktop settings UI can write/delete credentials, and the SSH execution layer
can look up the one credential for its already-validated destination.
"""

try:
    import gi

    gi.require_version("Secret", "1")
    from gi.repository import Secret
except (ImportError, ValueError):
    Secret = None


class SudoSecretError(RuntimeError):
    """A deliberately non-sensitive keyring error suitable for the UI."""


_SCHEMA = (
    Secret.Schema.new(
        "com.ronpatrick.Liam.SshSudo",
        Secret.SchemaFlags.NONE,
        {
            "application": Secret.SchemaAttributeType.STRING,
            "purpose": Secret.SchemaAttributeType.STRING,
            "alias": Secret.SchemaAttributeType.STRING,
            "hostname": Secret.SchemaAttributeType.STRING,
            "port": Secret.SchemaAttributeType.STRING,
            "user": Secret.SchemaAttributeType.STRING,
        },
    )
    if Secret is not None
    else None
)


def _attributes(alias, hostname, port, user):
    values = {
        "application": "com.ronpatrick.Liam",
        "purpose": "ssh-sudo",
        "alias": str(alias or "").strip(),
        "hostname": str(hostname or "").strip(),
        "port": str(port or "22").strip(),
        "user": str(user or "").strip(),
    }
    if not values["alias"] or not values["hostname"] or not values["user"]:
        raise SudoSecretError("The SSH destination is incomplete.")
    return values


def _require_keyring():
    if Secret is None or _SCHEMA is None:
        raise SudoSecretError("GNOME Keyring/libsecret is unavailable.")


def lookup_sudo_password(alias, hostname, port, user):
    """Return one destination's sudo password without exposing a model tool."""
    _require_keyring()
    try:
        return Secret.password_lookup_sync(
            _SCHEMA, _attributes(alias, hostname, port, user), None,
        )
    except Exception:
        raise SudoSecretError(
            "The sudo password could not be read from GNOME Keyring."
        ) from None


def has_sudo_password(alias, hostname, port, user):
    return bool(lookup_sudo_password(alias, hostname, port, user))


def store_sudo_password(alias, hostname, port, user, password):
    """Save or replace a destination's password directly through libsecret."""
    _require_keyring()
    if not isinstance(password, str) or not password:
        raise SudoSecretError("Enter a sudo password before saving.")
    if "\n" in password or "\r" in password or "\x00" in password:
        raise SudoSecretError("The sudo password cannot contain a line break.")
    attributes = _attributes(alias, hostname, port, user)
    label = f"Liam SSH sudo: {user}@{hostname}:{port} ({alias})"
    try:
        saved = Secret.password_store_sync(
            _SCHEMA, attributes, Secret.COLLECTION_DEFAULT,
            label, password, None,
        )
    except Exception:
        raise SudoSecretError(
            "The sudo password could not be saved in GNOME Keyring."
        ) from None
    if not saved:
        raise SudoSecretError(
            "The sudo password could not be saved in GNOME Keyring."
        )


def clear_sudo_password(alias, hostname, port, user):
    """Remove the exact destination credential, if it exists."""
    _require_keyring()
    try:
        return bool(Secret.password_clear_sync(
            _SCHEMA, _attributes(alias, hostname, port, user), None,
        ))
    except Exception:
        raise SudoSecretError(
            "The sudo password could not be removed from GNOME Keyring."
        ) from None
