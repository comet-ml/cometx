import importlib


def test_growth_report_delegate_exists():
    m = importlib.import_module("cometx.cli.admin_growth_report")
    assert hasattr(m, "generate_growth_report")


def test_growth_report_action_registered_in_admin():
    # admin.py builds an ACTION subparser; growth-report must be one of the choices
    from cometx.cli import admin as admin_mod

    src = admin_mod.__doc__ or ""  # noqa: F841
    # smoke: the delegate is imported by admin.py
    import inspect

    admin_src = inspect.getsource(admin_mod)
    assert "generate_growth_report" in admin_src
    assert "growth-report" in admin_src
