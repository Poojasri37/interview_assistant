from functools import wraps
from flask import session, redirect, url_for

def require_org_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("org_authed"):
            return redirect(url_for("org_login"))
        return fn(*args, **kwargs)
    return wrapper
