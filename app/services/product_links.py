"""Safe marketplace search links for non-chemical crop-care tools."""

from urllib.parse import quote_plus

SAFE_PRODUCT_TERMS = {
    "pruning shears": "pruning shears",
    "garden gloves": "garden gloves",
    "mulching film": "agricultural mulch film",
    "drip irrigation kit": "drip irrigation kit for small farms",
    "plant support clips": "plant support clips",
    "crop netting": "agricultural crop netting",
    "leaf moisture meter": "plant leaf moisture meter",
    "debris collection bags": "heavy duty garden waste bags",
}


def enrich_product_suggestions(suggestions: list) -> list:
    enriched = []

    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue

        item = dict(suggestion)
        category = " ".join(
            str(item.get("category", "")).lower()
            .replace("_", " ").split()
        )

        match = next(
            (
                term for term in SAFE_PRODUCT_TERMS
                if term in category or category in term
            ),
            None,
        )

        if match:
            query = quote_plus(SAFE_PRODUCT_TERMS[match])

            item["marketplace_links"] = [
                {
                    "marketplace": "Google Shopping",
                    "url": f"https://www.google.com/search?tbm=shop&q={query}",
                },
                {
                    "marketplace": "Amazon India",
                    "url": f"https://www.amazon.in/s?k={query}",
                },
                {
                    "marketplace": "Flipkart",
                    "url": f"https://www.flipkart.com/search?q={query}",
                },
            ]

            item["link_type"] = "marketplace_search"
            item["listing_disclaimer"] = (
                "Search links only; listings, sellers, prices and "
                "availability are not verified by Khetrakshak AI."
            )
        else:
            item["marketplace_links"] = []
            item["link_type"] = "not_linked"
            item["listing_disclaimer"] = (
                "No purchase link is provided for this category. "
                "Confirm chemical crop-protection products with a "
                "local agricultural expert and approved label."
            )

        enriched.append(item)

    return enriched