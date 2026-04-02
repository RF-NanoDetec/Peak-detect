"""Convert SVG logo to PNG with transparent background using Playwright."""
from playwright.sync_api import sync_playwright

# Read SVG content
with open("ui-web/public/logo-light.svg", "r", encoding="utf-8") as f:
    svg_content = f.read()

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    page.set_content(f"""
    <html>
    <body style="margin:0; padding:20px; background:transparent; display:flex; align-items:center; justify-content:center;">
        <div id="logo" style="width:1200px;">
            {svg_content}
        </div>
    </body>
    </html>
    """)
    page.wait_for_timeout(500)
    logo_el = page.query_selector("#logo")
    logo_el.screenshot(path="logo_light.png", omit_background=True)
    browser.close()
    print("Saved logo_light.png with transparent background")
