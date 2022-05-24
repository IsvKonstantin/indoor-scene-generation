COLORS = {
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'red': [255, 0, 0],
    'lime': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'silver': [192, 192, 192],
    'gray': [128, 128, 128],
    'maroon': [128, 0, 0],
    'olive': [128, 128, 0],
    'green': [0, 128, 0],
    'purple': [128, 0, 128],
    'teal': [0, 128, 128],
    'navy': [0, 0, 128]
}

COLOR_CODES = {
    (255, 0, 0): 1,     # lig
    (0, 0, 255): 2,     # ch
    (255, 0, 255): 3,   # tab
    (0, 255, 0): 4,     # cab
    (255, 255, 0): 5,   # sofa
    (128, 0, 128): 6    # bed
}

FURNITURE_COLORS = {
    'Cabinet/Shelf/Desk': 'lime',
    'Bed': 'purple',
    'Chair': 'blue',
    'Table': 'magenta',
    'Sofa': 'yellow',
    'Pier/Stool': 'cyan',
    'Lighting': 'red',
    'Others': 'green'
}

COLORS_ABOVE = {
    'purple': ['magenta', 'yellow', 'lime', 'blue', 'red'],
    'magenta': ['yellow', 'lime', 'blue', 'red'],
    'yellow': ['lime', 'blue', 'red'],
    'lime': ['blue', 'red'],
    'blue': ['red'],
    'cyan': [],
    'red': []
}

CONNECTION_CODES = {
    'void': 0,
    'floor': 1,
    'door': 2,
    'hole': 2,
    'window': 3
}

CONNECTION_COLORS = {
    'floor': 'silver',
    'hole': 'teal',
    'door': 'teal',
    'window': 'olive'
}

SUITABLE_ROOMS = {
    "LivingDiningRoom",
    "MasterBedroom",
    "SecondBedroom",
    "Bedroom",
    "LivingRoom",
    "DiningRoom",
}

CATEGORIES = {
    'Cabinet/Shelf/Desk': [
        'Children Cabinet',
        'Nightstand',
        'Bookcase / jewelry Armoire',
        'Wardrobe',
        'Coffee Table',
        'Corner/Side Table',
        'Sideboard / Side Cabinet / Console Table',
        'Wine Cabinet',
        'TV Stand',
        'Drawer Chest / Corner cabinet',
        'Shelf',
        'Round End Table',
        'Shoe Cabinet'
    ],
    'Bed': [
        'King-size Bed',
        'Bunk Bed',
        'Bed Frame',
        'Single bed',
        'Kids Bed',
        'Couch Bed'
    ],
    'Chair': [
        'Dining Chair',
        'Lounge Chair / Cafe Chair / Office Chair',
        'Dressing Chair',
        'Classic Chinese Chair',
        'Barstool',
        'Hanging Chair',
        'Folding chair',
    ],
    'Table': [
        'Dressing Table',
        'Dining Table',
        'Desk',
        'Bar',
    ],
    'Sofa': [
        'Three-Seat / Multi-seat Sofa',
        'armchair',
        'Loveseat Sofa',
        'L-shaped Sofa',
        'Lazy Sofa',
        'Chaise Longue Sofa',
        'U-shaped Sofa',
    ],
    'Pier/Stool': [
        'Footstool / Sofastool / Bed End Stool / Stool'
    ],
    'Lighting': [
        'Pendant Lamp',
        'Ceiling Lamp',
        'Floor Lamp',
        'Wall Lamp',
    ],
}

FILTER_CATEGORIES = {
    'Bunk Bed',
    'Floor Lamp',
    'Wall Lamp',
    'L-shaped Sofa',
    'Footstool / Sofastool / Bed End Stool / Stool'
}