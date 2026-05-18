# parts_database.py

# CRITICAL FIX: All dictionary lookup index keys forced to lowercase
AIRCRAFT_PARTS_REGISTRY = {
    "pn_623": [
        "623-12 (CAD PLATED)  -> AISLE 4, BIN B-12",
        "623-14 (TITANIUM)    -> AISLE 4, BIN B-14",
        "623-16 (OVERSIZE)    -> AISLE 5, BIN C-02"
    ],
    "pn_517": [
        "517-08 (SHEAR HEAD)  -> AISLE 2, BIN A-04",
        "517-12 (TENSILE)     -> AISLE 2, BIN A-08"
    ],
    "stapler": [
        "DESKTOP STATIONERY   -> AISLE 1, BIN S-01"
    ],
    "can": [
        "ALUMINUM CAN STORAGE -> RECYCLE BIN CO-1"
    ]
}

def get_bin_locations(family_name: str) -> list:
    """Queries your inventory file and returns shelf logs for the matched part."""
    # Force incoming strings to lowercase to ensure dictionary mapping alignment
    key = str(family_name).lower().strip()
    return AIRCRAFT_PARTS_REGISTRY.get(key, ["PART VALIDATED", "NO DATA IN INVENTORY TABLE"])