"""
data.py - Car catalogue for CarSense AI (Indian market).

NOTE: Prices (ex-showroom, Rs lakh), mileage, power and safety ratings are
approximate, rounded values compiled for ACADEMIC USE. Replace this table with
scraped / licensed data (CarDekho, CarWale, Bharat NCAP) for a production system.
"""
import pandas as pd

COLUMNS = ["name", "brand", "body", "ex_showroom", "fuel", "transmission", "seats",
           "mileage", "cc", "bhp", "boot_l", "ncap", "clearance", "battery_kwh", "ev_range"]

# mileage = kmpl (petrol/diesel/hybrid) or km/kg (CNG). EVs use battery_kwh + ev_range (km, claimed).
CARS = [
    ("Maruti Alto K10", "Maruti", "Hatchback", 4.5, "Petrol", "Manual", 5, 24.4, 998, 66, 214, 2, 160, 0, 0),
    ("Maruti S-Presso CNG", "Maruti", "Hatchback", 5.9, "CNG", "Manual", 5, 32.7, 998, 56, 240, 1, 165, 0, 0),
    ("Maruti Wagon R", "Maruti", "Hatchback", 6.0, "Petrol", "Manual", 5, 24.4, 1197, 89, 341, 2, 165, 0, 0),
    ("Maruti Celerio", "Maruti", "Hatchback", 6.2, "Petrol", "Manual", 5, 25.2, 998, 66, 313, 2, 170, 0, 0),
    ("Maruti Swift", "Maruti", "Hatchback", 7.0, "Petrol", "Manual", 5, 24.8, 1197, 80, 265, 3, 163, 0, 0),
    ("Maruti Baleno", "Maruti", "Hatchback", 7.5, "Petrol", "Manual", 5, 22.4, 1197, 89, 318, 3, 170, 0, 0),
    ("Maruti Dzire", "Maruti", "Sedan", 6.8, "Petrol", "Manual", 5, 24.8, 1197, 80, 382, 5, 163, 0, 0),
    ("Maruti Ciaz", "Maruti", "Sedan", 9.5, "Petrol", "Manual", 5, 20.6, 1462, 103, 510, 3, 170, 0, 0),
    ("Maruti Fronx", "Maruti", "Compact SUV", 7.5, "Petrol", "Manual", 5, 21.8, 1197, 89, 308, 4, 190, 0, 0),
    ("Maruti Brezza", "Maruti", "Compact SUV", 8.5, "Petrol", "Manual", 5, 19.8, 1462, 102, 328, 4, 198, 0, 0),
    ("Maruti Brezza CNG", "Maruti", "Compact SUV", 9.6, "CNG", "Manual", 5, 25.5, 1462, 87, 328, 4, 198, 0, 0),
    ("Maruti Grand Vitara", "Maruti", "Midsize SUV", 10.8, "Petrol", "Manual", 5, 21.1, 1462, 102, 373, 4, 210, 0, 0),
    ("Maruti Grand Vitara Hybrid", "Maruti", "Midsize SUV", 16.0, "Hybrid", "Automatic", 5, 27.9, 1490, 91, 373, 4, 210, 0, 0),
    ("Maruti Ertiga", "Maruti", "MPV", 9.0, "Petrol", "Manual", 7, 20.5, 1462, 102, 209, 3, 180, 0, 0),
    ("Maruti Ertiga CNG", "Maruti", "MPV", 10.3, "CNG", "Manual", 7, 26.1, 1462, 87, 209, 3, 180, 0, 0),
    ("Tata Tiago", "Tata", "Hatchback", 5.8, "Petrol", "Manual", 5, 20.0, 1199, 84, 242, 4, 170, 0, 0),
    ("Tata Tiago CNG", "Tata", "Hatchback", 7.0, "CNG", "Manual", 5, 26.5, 1199, 72, 242, 4, 170, 0, 0),
    ("Tata Tiago EV", "Tata", "Hatchback", 8.0, "Electric", "Automatic", 5, 0, 0, 74, 240, 4, 165, 19.2, 250),
    ("Tata Punch", "Tata", "Compact SUV", 6.2, "Petrol", "Manual", 5, 20.1, 1199, 87, 366, 5, 187, 0, 0),
    ("Tata Punch EV", "Tata", "Compact SUV", 10.9, "Electric", "Automatic", 5, 0, 0, 120, 366, 5, 190, 25.0, 315),
    ("Tata Altroz", "Tata", "Hatchback", 7.0, "Petrol", "Manual", 5, 19.3, 1199, 88, 345, 5, 165, 0, 0),
    ("Tata Altroz Diesel", "Tata", "Hatchback", 8.9, "Diesel", "Manual", 5, 23.6, 1497, 89, 345, 5, 165, 0, 0),
    ("Tata Nexon", "Tata", "Compact SUV", 8.2, "Petrol", "Manual", 5, 17.4, 1199, 118, 382, 5, 208, 0, 0),
    ("Tata Nexon Diesel", "Tata", "Compact SUV", 9.6, "Diesel", "Manual", 5, 23.2, 1497, 113, 382, 5, 208, 0, 0),
    ("Tata Nexon EV", "Tata", "Compact SUV", 12.5, "Electric", "Automatic", 5, 0, 0, 127, 350, 5, 205, 30.0, 325),
    ("Tata Curvv EV", "Tata", "Midsize SUV", 17.5, "Electric", "Automatic", 5, 0, 0, 167, 500, 5, 186, 45.0, 430),
    ("Tata Harrier", "Tata", "Midsize SUV", 15.0, "Diesel", "Manual", 5, 16.4, 1956, 168, 445, 5, 205, 0, 0),
    ("Tata Safari", "Tata", "Midsize SUV", 16.2, "Diesel", "Manual", 7, 16.3, 1956, 168, 420, 5, 205, 0, 0),
    ("Hyundai Grand i10 Nios", "Hyundai", "Hatchback", 6.0, "Petrol", "Manual", 5, 20.7, 1197, 83, 260, 2, 165, 0, 0),
    ("Hyundai i20", "Hyundai", "Hatchback", 7.5, "Petrol", "Manual", 5, 20.3, 1197, 83, 311, 3, 170, 0, 0),
    ("Hyundai Exter", "Hyundai", "Compact SUV", 6.2, "Petrol", "Manual", 5, 19.4, 1197, 82, 391, 4, 185, 0, 0),
    ("Hyundai Aura", "Hyundai", "Sedan", 6.5, "Petrol", "Manual", 5, 20.0, 1197, 82, 402, 2, 165, 0, 0),
    ("Hyundai Venue", "Hyundai", "Compact SUV", 8.0, "Petrol", "Manual", 5, 18.0, 1197, 82, 350, 3, 190, 0, 0),
    ("Hyundai Verna", "Hyundai", "Sedan", 11.0, "Petrol", "Manual", 5, 18.6, 1497, 113, 528, 5, 165, 0, 0),
    ("Hyundai Creta", "Hyundai", "Midsize SUV", 11.1, "Petrol", "Manual", 5, 17.4, 1497, 113, 433, 5, 190, 0, 0),
    ("Hyundai Creta Diesel", "Hyundai", "Midsize SUV", 12.5, "Diesel", "Manual", 5, 21.8, 1493, 114, 433, 5, 190, 0, 0),
    ("Kia Sonet", "Kia", "Compact SUV", 8.0, "Petrol", "Manual", 5, 18.4, 1197, 82, 385, 3, 205, 0, 0),
    ("Kia Seltos", "Kia", "Midsize SUV", 11.1, "Petrol", "Manual", 5, 17.0, 1482, 113, 433, 5, 190, 0, 0),
    ("Kia Carens", "Kia", "MPV", 11.0, "Petrol", "Manual", 7, 16.0, 1497, 113, 216, 3, 185, 0, 0),
    ("Honda Amaze", "Honda", "Sedan", 7.5, "Petrol", "Manual", 5, 18.6, 1199, 89, 416, 4, 170, 0, 0),
    ("Honda City", "Honda", "Sedan", 12.0, "Petrol", "Manual", 5, 17.8, 1498, 119, 506, 4, 165, 0, 0),
    ("Honda Elevate", "Honda", "Midsize SUV", 11.7, "Petrol", "Manual", 5, 15.3, 1498, 119, 458, 4, 220, 0, 0),
    ("Mahindra XUV 3XO", "Mahindra", "Compact SUV", 7.5, "Petrol", "Manual", 5, 18.9, 1197, 109, 364, 5, 201, 0, 0),
    ("Mahindra Thar", "Mahindra", "Midsize SUV", 11.5, "Diesel", "Manual", 4, 15.2, 2184, 130, 300, 5, 226, 0, 0),
    ("Mahindra Scorpio-N", "Mahindra", "Large SUV", 13.6, "Diesel", "Manual", 7, 14.4, 2198, 172, 460, 5, 187, 0, 0),
    ("Mahindra XUV700", "Mahindra", "Large SUV", 14.0, "Diesel", "Manual", 7, 16.4, 2184, 182, 240, 5, 200, 0, 0),
    ("Mahindra XUV400 EV", "Mahindra", "Compact SUV", 15.5, "Electric", "Automatic", 5, 0, 0, 148, 378, 5, 200, 39.4, 456),
    ("Toyota Innova Crysta", "Toyota", "MPV", 19.99, "Diesel", "Manual", 7, 15.0, 2393, 148, 300, 4, 178, 0, 0),
    ("Toyota Innova Hycross", "Toyota", "MPV", 19.8, "Hybrid", "Automatic", 7, 23.2, 1987, 183, 300, 5, 185, 0, 0),
    ("Toyota Hyryder", "Toyota", "Midsize SUV", 11.0, "Petrol", "Manual", 5, 19.4, 1462, 102, 373, 4, 210, 0, 0),
    ("Toyota Hyryder Hybrid", "Toyota", "Midsize SUV", 16.5, "Hybrid", "Automatic", 5, 27.9, 1490, 91, 373, 4, 210, 0, 0),
    ("Toyota Fortuner", "Toyota", "Large SUV", 35.0, "Diesel", "Manual", 7, 14.2, 2755, 201, 296, 5, 225, 0, 0),
    ("VW Virtus", "Volkswagen", "Sedan", 11.5, "Petrol", "Manual", 5, 19.4, 999, 114, 521, 5, 179, 0, 0),
    ("VW Taigun", "Volkswagen", "Midsize SUV", 11.7, "Petrol", "Manual", 5, 19.2, 999, 114, 385, 5, 188, 0, 0),
    ("Skoda Slavia", "Skoda", "Sedan", 11.0, "Petrol", "Manual", 5, 19.4, 999, 114, 521, 5, 179, 0, 0),
    ("Skoda Kushaq", "Skoda", "Midsize SUV", 11.0, "Petrol", "Manual", 5, 19.1, 999, 114, 385, 5, 188, 0, 0),
    ("Nissan Magnite", "Nissan", "Compact SUV", 6.3, "Petrol", "Manual", 5, 19.4, 999, 72, 336, 4, 205, 0, 0),
    ("Renault Triber", "Renault", "MPV", 6.2, "Petrol", "Manual", 7, 18.3, 999, 72, 84, 4, 182, 0, 0),
    ("MG Hector", "MG", "Midsize SUV", 14.0, "Petrol", "Manual", 5, 15.3, 1451, 141, 587, 4, 192, 0, 0),
    ("MG Windsor EV", "MG", "Midsize SUV", 14.0, "Electric", "Automatic", 5, 0, 0, 134, 604, 4, 186, 38.0, 331),
    ("MG Comet EV", "MG", "Hatchback", 7.0, "Electric", "Automatic", 4, 0, 0, 41, 200, 3, 165, 17.3, 230),
]

# Brand reliability / resale-trust score (0-1) - encodes market reputation for resale and service network.
BRAND_SCORE = {"Toyota": 0.90, "Maruti": 0.85, "Hyundai": 0.75, "Honda": 0.72, "Mahindra": 0.72,
               "Kia": 0.68, "Tata": 0.62, "Volkswagen": 0.60, "Skoda": 0.60, "MG": 0.50,
               "Nissan": 0.50, "Renault": 0.50}

# Relative service-cost level of each brand.
BRAND_SERVICE = {"Maruti": 0.80, "Toyota": 0.90, "Hyundai": 1.00, "Tata": 1.00, "Renault": 1.00,
                 "Nissan": 1.00, "Kia": 1.10, "Honda": 1.10, "Mahindra": 1.15, "MG": 1.20,
                 "Volkswagen": 1.40, "Skoda": 1.40}

CARS_DF = pd.DataFrame(CARS, columns=COLUMNS)
CARS_DF["brand_score"] = CARS_DF["brand"].map(BRAND_SCORE)
BODY_TYPES = sorted(CARS_DF["body"].unique())
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"]

if __name__ == "__main__":
    print(CARS_DF.shape)
    print(CARS_DF.head())
