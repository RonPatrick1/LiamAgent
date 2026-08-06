"""GNOME Keyring storage for desktop sudo credentials (SSH destinations and
this machine's own local sudo).

This module is intentionally not registered as an agent tool.  Only the
desktop settings UI can write/delete credentials, and the SSH/local execution
layers can look up the one credential for their already-validated target.
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

# Separate schema from SSH's (different, smaller attribute set — this
# machine, not a remote destination) so the two credential kinds can never
# collide or be looked up with each other's key.
_LOCAL_SCHEMA = (
    Secret.Schema.new(
        "com.ronpatrick.Liam.LocalSudo",
        Secret.SchemaFlags.NONE,
        {
            "application": Secret.SchemaAttributeType.STRING,
            "purpose": Secret.SchemaAttributeType.STRING,
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


def _local_attributes(user):
    values = {
        "application": "com.ronpatrick.Liam",
        "purpose": "local-sudo",
        "user": str(user or "").strip(),
    }
    if not values["user"]:
        raise SudoSecretError("The local user is unknown.")
    return values


def lookup_local_sudo_password(user):
    """Return this machine's stored local sudo password, or None."""
    _require_keyring()
    try:
        return Secret.password_lookup_sync(
            _LOCAL_SCHEMA, _local_attributes(user), None,
        )
    except Exception:
        raise SudoSecretError(
            "The local sudo password could not be read from GNOME Keyring."
        ) from None


def has_local_sudo_password(user):
    return bool(lookup_local_sudo_password(user))


def store_local_sudo_password(user, password):
    """Save or replace this machine's local sudo password."""
    _require_keyring()
    if not isinstance(password, str) or not password:
        raise SudoSecretError("Enter a sudo password before saving.")
    if "\n" in password or "\r" in password or "\x00" in password:
        raise SudoSecretError("The sudo password cannot contain a line break.")
    attributes = _local_attributes(user)
    label = f"Liam local sudo: {user}"
    try:
        saved = Secret.password_store_sync(
            _LOCAL_SCHEMA, attributes, Secret.COLLECTION_DEFAULT,
            label, password, None,
        )
    except Exception:
        raise SudoSecretError(
            "The local sudo password could not be saved in GNOME Keyring."
        ) from None
    if not saved:
        raise SudoSecretError(
            "The local sudo password could not be saved in GNOME Keyring."
        )


def clear_local_sudo_password(user):
    """Remove this machine's local sudo credential, if it exists."""
    _require_keyring()
    try:
        return bool(Secret.password_clear_sync(
            _LOCAL_SCHEMA, _local_attributes(user), None,
        ))
    except Exception:
        raise SudoSecretError(
            "The local sudo password could not be removed from GNOME Keyring."
        ) from None
