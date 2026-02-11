# agents/booking_agent.py
"""
Merged BookingAgent:
 - Flights: Amadeus test API (v2/shopping/flight-offers) + OAuth token caching
 - Other helpers (hotels, buses, trains, movies, IATA fuzzy lookup)
 - Conversational booking flow
 - Sandbox payment workflow (simulated payment URL opening)
"""

import re
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus, urlencode
import webbrowser
import os
import dateparser
import requests

from config import (
    AMADEUS_API_KEY,
    AMADEUS_API_SECRET,
    AMADEUS_BASE_URL,
)
from utils import open_with, speak

_AMADEUS_BASE = AMADEUS_BASE_URL.rstrip("/") if AMADEUS_BASE_URL else "https://test.api.amadeus.com"


class BookingAgent:
    """
    Booking/search + conversational booking agent using Amadeus for flights,
    and URL helpers for buses/trains/hotels/movies.
    """

    # ---------- data ----------
    SUPPORTED_LANGS = {
        "af", "bs", "ca", "cs", "da", "de", "et", "en-GB", "en-US", "es", "es-419",
        "eu", "fil", "fr", "gl", "hr", "id", "is", "it", "sw", "lv", "lt", "hu",
        "ms", "nl", "no", "pl", "pt-BR", "pt-PT", "ro", "sq", "sk", "sl", "sr-Latn",
        "fi", "sv", "vi", "tr", "el", "bg", "mk", "mn", "ru", "sr", "uk", "ka",
        "iw", "ur", "ar", "fa", "am", "ne", "mr", "hi", "bn", "pa", "gu", "ta",
        "te", "kn", "ml", "si", "th", "lo", "km", "ko", "ja", "zh-CN", "zh-TW",
    }

    IATA = {
        "delhi": "DEL",
        "new delhi": "DEL",
        "del": "DEL",
        "mumbai": "BOM",
        "bombay": "BOM",
        "bom": "BOM",
        "bengaluru": "BLR",
        "bangalore": "BLR",
        "blr": "BLR",
        "chennai": "MAA",
        "madras": "MAA",
        "maa": "MAA",
        "kolkata": "CCU",
        "calcutta": "CCU",
        "ccu": "CCU",
        "hyderabad": "HYD",
        "hyd": "HYD",
        "ahmedabad": "AMD",
        "amd": "AMD",
        "jaipur": "JAI",
        "jai": "JAI",
        "srinagar": "SXR",
        "sxr": "SXR",
        "goa": "GOI",
        "goi": "GOI",
    }

    CITY_CORRECTIONS = {
        "sinehagar": "srinagar",
        "sinegar": "srinagar",
        "srinigar": "srinagar",
        "srinager": "srinagar",
    }

    def __init__(self):
        # last_results is generic:
        #  - flights: dict with offer, price, etc. (no "type" key)
        #  - buses/trains/etc: {"type": "bus", "index": n, "url": "...", ...}
        self.last_results: List[Dict[str, Any]] = []
        self.airport_cache: Dict[str, str] = {}
        self.booking_state: Dict[str, Any] = {
            "active": False,
            "step": None,
            "index": None,
            "flight": None,
            "data": {},
        }
        self._token_cache: Dict[str, Any] = {}  # for Amadeus token caching

    # ---------- booking state ----------
    def in_booking(self) -> bool:
        return bool(self.booking_state.get("active"))

    # ---------- small helper for "from X to Y" ----------
    def _extract_from_to(self, text: str):
        """
        Extract 'from X to Y' cities from a sentence.
        Used especially for bus queries like:
          'for bus from Delhi to Bangalore'
        Returns (from_city, to_city) or (None, None).
        """
        t = text.strip().lower().rstrip(".?!")
        t = re.sub(r"^\s*(for\s+bus|bus|for)\s+", "", t)

        m = re.search(r"\bfrom\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)\b", t, re.IGNORECASE)
        if not m:
            return None, None
        from_city = m.group(1).strip()
        to_city = m.group(2).strip()
        return from_city, to_city

    # ---------- entry point ----------
    def handle(self, cmd: Dict[str, Any]):
        text = (cmd.get("original_text") or "").strip()
        norm = (cmd.get("normalized") or {}) or {}
        t_lower = text.lower()

        # If mid-booking, continue
        if self.in_booking():
            self._continue_booking(text)
            return

        # Quick commands (book/open option)
        book_idx = self._parse_book_option_index(text)
        if book_idx is not None:
            self._start_booking_for_option(book_idx)
            return

        open_idx = self._parse_open_option_index(text)
        if open_idx is not None:
            self._open_option(open_idx)
            return

        # Parse cities/dates/mode
        parts = self._find_city_tokens(text)
        depart_iso, ret_iso = self._parse_when(text, norm)

        mode = parts["mode"]
        origin = parts["origin"]
        dest = parts["destination"]
        city = parts["city"]

        # For bus queries, refine origin/dest using the special extractor
        if mode == "bus":
            fb_from, fb_to = self._extract_from_to(text)
            if fb_from and fb_to:
                origin, dest = fb_from, fb_to

        # If nothing explicit but origin+dest present, assume flights
        if not mode and origin and dest:
            if "flight" in t_lower or "flights" in t_lower or "cheapest" in t_lower:
                mode = "flights"

        # Flights -> Amadeus
        if mode == "flights" and origin and dest:
            if not depart_iso:
                depart_iso = datetime.now().date().isoformat()
            self._search_amadeus_flights(origin, dest, depart_iso, ret_iso)
            return

        # Other modes -> URL helpers
        urls: List[str] = []
        if mode == "bus" and origin and dest:
            urls = self._mk_bus_urls(origin, dest, depart_iso)
        elif mode == "train" and origin and dest:
            urls = self._mk_train_urls(origin, dest, depart_iso)
        elif mode == "movie":
            m = re.search(r'["“](.+?)["”]', text)
            title = m.group(1) if m else None
            movie_city = city or origin or dest or "mumbai"
            urls = self._mk_movie_urls(movie_city, title, depart_iso)
        elif mode == "hotel":
            hotel_city = city or origin or dest
            if hotel_city:
                urls = self._mk_hotel_urls(hotel_city, depart_iso, ret_iso)

        if not urls:
            urls = ["https://www.google.com/search?q=" + quote_plus(text)]

        # ---- SPECIAL HANDLING FOR BUS: show options, don't auto-open ----
        if mode == "bus" and origin and dest:
            self.last_results = []
            print("(booking) found bus options:")

            for idx, u in enumerate(urls, start=1):
                if "makemytrip.com" in u:
                    title = f"Bus results on MakeMyTrip: {origin.title()} → {dest.title()}"
                elif "google.com" in u:
                    title = f"Google search: buses from {origin} to {dest}"
                else:
                    title = f"Option {idx}: {u}"

                self.last_results.append({
                    "type": "bus",
                    "index": idx,
                    "title": title,
                    "url": u,
                    "from": origin,
                    "to": dest,
                })
                print(f"  {idx}. {title}\n     {u}")

            speak("I found some bus options. Say 'open option 1' or 'open option 2'.")
            return

        # ---- Default behaviour for trains/movies/hotels/others: auto-open ----
        print("(booking) opening:")
        for u in urls:
            print(" -", u)
        self._open_urls(urls)

    # ---------- option index parsing ----------
    def _parse_open_option_index(self, text: str) -> Optional[int]:
        t = text.lower()
        m = re.search(r"\bopen\s+(?:the\s+)?option\s+(\d+)(?:st|nd|rd|th)?\b", t)
        if m:
            return int(m.group(1))
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        m_word_opt = re.search(r"\bopen\s+(?:the\s+)?option\s+(one|two|three|four|five)\b", t)
        if m_word_opt:
            return word_to_num[m_word_opt.group(1)]
        m = re.search(r"\bopen\s+(\d+)(?:st|nd|rd|th)?\b", t)
        if m:
            return int(m.group(1))
        ord_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        m_ord = re.search(r"\bopen\s+(?:the\s+)?(first|second|third|fourth|fifth)\s+(?:option|one)\b", t)
        if m_ord:
            return ord_map[m_ord.group(1)]
        return None

    def _parse_book_option_index(self, text: str) -> Optional[int]:
        t = text.lower()
        m = re.search(r"\bbook\s+(?:the\s+)?option\s+(\d+)(?:st|nd|rd|th)?\b", t)
        if m:
            return int(m.group(1))
        ord_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        m = re.search(r"\bbook\s+(?:the\s+)?(first|second|third|fourth|fifth)\s+(?:one|option)?\b", t)
        if m:
            return ord_map[m.group(1)]
        m = re.search(r"\bbook\s+(\d+)(?:st|nd|rd|th)?\b", t)
        if m:
            return int(m.group(1))
        return None

    # ---------- booking flow ----------
    def _start_booking_for_option(self, index: int):
        if not self.last_results:
            speak("I don't have any options cached. Ask me to search flights first.")
            return
        if index < 1 or index > len(self.last_results):
            speak("That option number is out of range.")
            return

        opt = self.last_results[index - 1]

        # Booking flow is only for flights (no 'type' key)
        # Booking flow is only for flights
        if opt.get("type") and opt.get("type") != "flight":
            speak("Booking flow is only implemented for flights, not buses yet.")
            return


        flight = opt
        summary = (
            f"{flight.get('from', '')} to {flight.get('to', '')}, "
            f"departing {flight.get('depart', '-')}, "
            f"price {flight.get('price_str', 'N/A')}, "
            f"{flight.get('stops', 0)} stop"
            + ("" if flight.get('stops', 0) == 1 else "s")
        )

        self.booking_state = {
            "active": True,
            "step": "ask_name",
            "index": index,
            "flight": flight,
            "data": {},
        }

        print(f"(booking) Starting booking for option {index}: {summary}")
        speak(
            f"Booking option {index}: {summary}. "
            f"Let's fill in the passenger details. What is the passenger's full name?"
        )

    def _continue_booking(self, text: str):
        t = text.lower().strip()
        st = self.booking_state.get("step")
        data = self.booking_state.get("data", {})

        # ---- cancel handling ----
        if any(w in t for w in ["cancel", "stop booking", "abort booking", "never mind", "nevermind"]):
            speak("Okay, I've cancelled the booking flow.")
            self.booking_state = {
                "active": False,
                "step": None,
                "index": None,
                "flight": None,
                "data": {},
            }
            return

        def is_yes(s: str) -> bool:
            return any(w in s for w in ["yes", "yeah", "yep", "sure", "confirm", "go ahead", "ok", "okay"])

        def is_no(s: str) -> bool:
            return any(w in s for w in ["no", "nope", "don't", "dont", "not now", "later", "stop"])

        # ---- STEP 1: NAME ----
        if st == "ask_name":
            if len(text.split()) < 2:
                speak("Please say the full name, like John Doe.")
                return

            data["name"] = text.strip()
            self.booking_state["step"] = "ask_age"
            speak(f"Got it. Passenger name is {data['name']}. What is the age?")
            return

        # ---- STEP 2: AGE ----
        if st == "ask_age":
            m = re.search(r"\d+", t)
            if not m:
                speak("Please tell me the age in numbers.")
                return

            age = int(m.group(0))
            if age <= 0 or age > 120:
                speak("That age doesn't look right. Please say it again.")
                return

            data["age"] = age
            self.booking_state["step"] = "ask_gender"
            speak("Noted. What is the passenger's gender?")
            return

        # ---- STEP 3: GENDER + AUTO BOOKING DATA ----
        if st == "ask_gender":
            if any(g in t for g in ["male", "man", "m"]):
                data["gender"] = "Male"
            elif any(g in t for g in ["female", "woman", "f"]):
                data["gender"] = "Female"
            elif any(g in t for g in ["other", "non-binary", "nonbinary"]):
                data["gender"] = "Other"
            else:
                speak("Please say male, female, or other.")
                return

            # ---- generate realistic booking metadata ----
            data["pnr"] = self._generate_pnr()
            data["ticket_no"] = self._generate_ticket_number()
            data["booked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            self.booking_state["step"] = "confirm"

            f = self.booking_state.get("flight", {})
            summary = (
                f"Flight {self.booking_state['index']} "
                f"{f.get('from','')} to {f.get('to','')}, "
                f"depart {f.get('depart','-')}, "
                f"price {f.get('price_str','N/A')}. "
                f"Passenger {data['name']} ({data['age']} yrs, {data['gender']}). "
                f"PNR {data['pnr']}, Ticket {data['ticket_no']}."
            )

            print("(booking) Summary:", summary)
            speak(
                summary +
                " If this is correct, say confirm to proceed to payment, "
                "or say no to cancel."
            )
            return

        # ---- STEP 4: CONFIRM ----
        if st == "confirm":
            if is_yes(t):
                self.booking_state["step"] = "payment"
                self._start_payment()
                return

            if is_no(t):
                speak("Okay, cancelled the booking.")
                self.booking_state = {
                    "active": False,
                    "step": None,
                    "index": None,
                    "flight": None,
                    "data": {},
                }
                return

            speak("Please say confirm to proceed, or no to cancel.")
            return

        # ---- PAYMENT ----
        if st == "payment":
            speak("Payment is already in progress. Please complete it in your browser.")
            return

        speak("I didn't catch that. Please answer the current question or say cancel.")


    # ---------- payment workflow (sandbox) ----------
    def _start_payment(self):
        index = self.booking_state.get("index")
        if not index or not self.last_results:
            speak("I lost the selected flight details. Please search again.")
            self.booking_state = {"active": False, "step": None, "index": None, "flight": None, "data": {}}
            return

        opt = self.last_results[index - 1]
        offer = opt.get("offer") or {}
        offer_id = offer.get("id") or offer.get("offerId") or f"sandbox-{int(time.time())}-{index}"

        price_obj = (offer.get("price") or {})
        amount = price_obj.get("total") or price_obj.get("grandTotal") or opt.get("price") or "N/A"
        currency = price_obj.get("currency") or (opt.get("price_str") or "").split()[-1] if opt.get("price_str") else "INR"

        payment_params = {
            "offer_id": offer_id,
            "amount": amount,
            "currency": currency,
            "passenger_name": self.booking_state["data"]["name"],

            # ✅ new realistic booking fields
            "pnr": self.booking_state["data"]["pnr"],
            "ticket_no": self.booking_state["data"]["ticket_no"],
            "booked_at": self.booking_state["data"]["booked_at"],
        }


        qs = urlencode(payment_params, quote_via=quote_plus)
        local_path = "http://localhost:8000/sandbox/pay.html"
        sandbox_payment_url = f"{local_path}?{qs}"

        print(f"(booking) Sandbox payment URL: {sandbox_payment_url}")

        speak("Great. I'll open a sandbox payment page in your browser so you can complete the simulated payment.")
        try:
            opened = webbrowser.open(sandbox_payment_url, new=2)
            if opened:
                speak("Opened the sandbox payment page in your browser (simulated). Complete the process there.")
            else:
                ok = open_with(["firefox", "google-chrome", "chromium-browser", "xdg-open"], [sandbox_payment_url])
                if ok:
                    speak("Opened the sandbox payment page in your browser (simulated). Complete the process there.")
                else:
                    speak("I couldn't open the browser. The sandbox payment URL is printed in the console.")
                    print(sandbox_payment_url)
        except Exception as e:
            print("(booking) error opening sandbox URL:", e)
            speak("I couldn't open the browser. The sandbox payment URL is printed in the console.")
            print(sandbox_payment_url)

        self.booking_state = {"active": False, "step": None, "index": None, "flight": None, "data": {}}

    # ---------- date parsing ----------
    def _parse_when(self, text: str, norm: Dict[str, Any] | None = None):
        t = text.lower()
        today = datetime.now().date()

        m_rng = re.search(r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*([a-z]{3,})\b", t)
        if m_rng:
            d1, d2, mon = m_rng.groups()
            d1 = dateparser.parse(f"{d1} {mon}", settings={"PREFER_DATES_FROM": "future"})
            d2 = dateparser.parse(f"{d2} {mon}", settings={"PREFER_DATES_FROM": "future"})
            if d1 and d2:
                return d1.date().isoformat(), d2.date().isoformat()

        m_fromto = re.search(r"\bfrom\s+([a-z0-9 \-\/]+?)\s+(?:to|till|until)\s+([a-z0-9 \-\/]+)\b", t)
        if m_fromto:
            d1 = dateparser.parse(m_fromto.group(1), settings={"PREFER_DATES_FROM": "future"})
            d2 = dateparser.parse(m_fromto.group(2), settings={"PREFER_DATES_FROM": "future"})
            if d1 and d2:
                return d1.date().isoformat(), d2.date().isoformat()

        m_dm = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", t)
        if m_dm:
            day = m_dm.group(1)
            mon = m_dm.group(3)
            dt = dateparser.parse(f"{day} {mon}", settings={"PREFER_DATES_FROM": "future"})
            if dt:
                return dt.date().isoformat(), None

        if "day after tomorrow" in t:
            return (today + timedelta(days=2)).isoformat(), None
        if "tomorrow" in t:
            return (today + timedelta(days=1)).isoformat(), None
        if "today" in t:
            return today.isoformat(), None

        if norm:
            dt_val = norm.get("datetime")
            if dt_val:
                try:
                    from dateutil import parser as dtparse
                    dt = dtparse.isoparse(dt_val)
                    return dt.date().isoformat(), None
                except Exception:
                    pass

        dt = dateparser.parse(t, settings={"PREFER_DATES_FROM": "future"})
        if dt:
            return dt.date().isoformat(), None

        return None, None

    # ---------- city parsing ----------
    def _find_city_tokens(self, text: str):
        t = text.lower()
        mode = None

        if "flight" in t or "flights" in t:
            mode = "flights"
        elif "bus" in t or "buses" in t:
            mode = "bus"
        elif "train" in t or "trains" in t:
            mode = "train"
        elif any(k in t for k in ["movie", "movies", "cinema", "theatre", "tickets"]):
            mode = "movie"
        elif any(k in t for k in ["hotel", "hotels", "stay"]):
            mode = "hotel"

        origin = dest = city = None

        m1 = re.search(r"\bfrom\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+?)(?:\s+(?:on|for|at)\b|$)", t)
        if m1:
            origin = m1.group(1).strip()
            dest = m1.group(2).strip()
        else:
            m2 = re.search(r"\b([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+)\b", t)
            if m2:
                origin = m2.group(1).strip()
                dest = m2.group(2).strip()

        m_in = re.search(r"\bin\s+([a-zA-Z ]+)\b", t)
        if m_in:
            city = m_in.group(1).strip()

        return {"mode": mode, "origin": origin, "destination": dest, "city": city}

    # ---------- helpers ----------
    def _slug(self, s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", s.lower().strip())
        return re.sub(r"-+", "-", s).strip("-")

    def _slug_city(self, name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return re.sub(r"-+", "-", s).strip("-")

    def _open_urls(self, urls: List[str]):
        any_ok = False
        for url in urls:
            ok = open_with(["firefox", "google-chrome", "chromium-browser", "xdg-open"], [url])
            any_ok = any_ok or ok
        if any_ok:
            speak("Opened results in your browser.")
        else:
            speak("I couldn't open the browser. URLs are printed in the console.")
            for u in urls:
                print(u)

    def _normalize_city_key(self, term: str) -> str:
        key = term.strip().lower()
        if key in self.CITY_CORRECTIONS:
            return self.CITY_CORRECTIONS[key]
        return key

    def _iata(self, name: Optional[str]):
        if not name:
            return None
        key = self._normalize_city_key(name)
        if key in self.IATA:
            return self.IATA[key]
        try:
            from rapidfuzz import process as rf_process, fuzz as rf_fuzz
            cand = rf_process.extractOne(key, list(self.IATA.keys()), scorer=rf_fuzz.WRatio)
            if cand and len(cand) >= 2:
                candidate_name, score, _ = cand
                if score >= 80:
                    return self.IATA.get(candidate_name)
        except Exception:
            pass
        return None
    
    def _generate_pnr(self) -> str:
        import random, string
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    def _generate_ticket_number(self) -> str:
        import random
        return f"{random.randint(100,999)}-{random.randint(1000000000,9999999999)}"


    # ---------- URL helpers ----------
    def _mk_flight_urls(self, origin, dest, depart=None, ret=None):
        o = self._iata(origin) or origin
        d = self._iata(dest) or dest
        q = f"flights from {o} to {d}"
        if depart:
            q += f" on {depart}"
        if ret:
            q += f" to {ret}"
        return ["https://www.google.com/travel/flights?q=" + quote_plus(q)]
    
    def _make_flight_booking_url(self, opt: Dict[str, Any]) -> str:
        """
        Canonical booking URL for a flight option.
        Opens a real, working booking page.
        """
        o = opt["from"]
        d = opt["to"]
        dep = (opt.get("depart") or "").split("T")[0]

        query = f"{o} to {d} on {dep}"
        return "https://www.google.com/travel/flights?q=" + quote_plus(query)



    def _mk_bus_urls(self, origin, dest, depart=None):
        o_slug = self._slug_city(origin)
        d_slug = self._slug_city(dest)

        mmt = (
            f"https://www.makemytrip.com/bus-tickets/"
            f"{o_slug}-{d_slug}-bus-ticket-booking.html"
        )

        q = f"buses from {origin} to {dest}"
        if depart:
            q += f" on {depart}"
        g = "https://www.google.com/search?q=" + quote_plus(q)

        return [mmt, g]

    def _mk_train_urls(self, origin, dest, depart=None):
        q = f"trains from {origin} to {dest}"
        if depart:
            q += f" {depart}"
        g = "https://www.google.com/search?q=" + quote_plus(q)
        ixigo = "https://www.ixigo.com/trains/" + quote_plus(f"{origin}-to-{dest}")
        return [g, ixigo]

    def _mk_movie_urls(self, city, title=None, when=None):
        bms_city = f"https://in.bookmyshow.com/explore/movies-{self._slug(city)}"
        if title:
            q = f"bookmyshow {city} {title}"
            if when:
                q += f" {when}"
            g = "https://www.google.com/search?q=" + quote_plus(q)
            return [bms_city, g]
        return [bms_city]

    def _mk_hotel_urls(self, city, checkin=None, checkout=None):
        q = f"hotels in {city}"
        if checkin:
            q += f" from {checkin}"
        if checkout:
            q += f" to {checkout}"
        gh = "https://www.google.com/travel/hotels?q=" + quote_plus(q)
        mmt = "https://www.makemytrip.com/hotels/" + self._slug(city) + "-hotels.html"
        return [gh, mmt]

    # ---------- Amadeus helpers ----------
    def _get_amadeus_token(self) -> Optional[str]:
        now_ts = int(time.time())
        cached = self._token_cache.get("access_token")
        expires_at = self._token_cache.get("expires_at", 0)
        if cached and expires_at > now_ts + 10:
            return cached

        if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
            print("(booking) Amadeus credentials not configured.")
            return None

        url = f"{_AMADEUS_BASE}/v1/security/oauth2/token"
        data = {"grant_type": "client_credentials", "client_id": AMADEUS_API_KEY, "client_secret": AMADEUS_API_SECRET}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            r = requests.post(url, data=data, headers=headers, timeout=10)
            r.raise_for_status()
            j = r.json()
            token = j.get("access_token")
            expires_in = int(j.get("expires_in", 0) or 0)
            if token:
                self._token_cache["access_token"] = token
                self._token_cache["expires_at"] = now_ts + expires_in
                return token
            print("(booking) Amadeus token response missing token:", j)
        except Exception as e:
            print("(booking) Amadeus token fetch error:", e)
        return None

    def _search_amadeus_flights(self, origin: str, dest: str, depart_iso: str, ret_iso: Optional[str] = None):
        token = self._get_amadeus_token()
        if not token:
            speak("Amadeus credentials missing or token fetch failed. Opening Google Flights instead.")
            self._open_urls(self._mk_flight_urls(origin, dest, depart_iso, ret_iso))
            return

        o_code = self._iata(origin) or (origin[:3].upper())
        d_code = self._iata(dest) or (dest[:3].upper())

        url = f"{_AMADEUS_BASE}/v2/shopping/flight-offers"
        params = {
            "originLocationCode": o_code,
            "destinationLocationCode": d_code,
            "departureDate": depart_iso,
            "adults": 1,
            "currencyCode": "INR",
            "max": 10,
        }
        if ret_iso:
            params["returnDate"] = ret_iso

        headers = {"Authorization": f"Bearer {token}"}

        print(f"(booking) Amadeus flight-offers → {url}")
        print("(booking) Params:", params)

        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("(booking) Amadeus search error:", e)
            speak("Flight search failed via Amadeus; opening Google Flights as fallback.")
            self._open_urls(self._mk_flight_urls(origin, dest, depart_iso, ret_iso))
            return

        offers = data.get("data", []) or []
        if not offers:
            print("(booking) No offers from Amadeus:", data)
            speak("No flights found for that route and date.")
            return

        self.last_results = []
        rows = []
        for i, offer in enumerate(offers[:10], start=1):
            try:
                itineraries = offer.get("itineraries", [])
                first_itin = itineraries[0] if itineraries else {}
                segments = first_itin.get("segments", []) if first_itin else []
                first_seg = segments[0] if segments else {}
                airline = first_seg.get("carrierCode")
                flight_no = first_seg.get("number")

                dep = segments[0].get("departure", {}).get("at") if segments else None
                arr = segments[-1].get("arrival", {}).get("at") if segments else None
                duration = first_itin.get("duration", "")

                price_obj = offer.get("price") or {}
                price_total = price_obj.get("total") or price_obj.get("grandTotal") or price_obj.get("totalPrice")
                currency = price_obj.get("currency") or "USD"
                price_str = f"{price_total} {currency}" if price_total else "N/A"
                stops = max(0, len(segments) - 1)
                carrier_codes = []
                for seg in segments:
                    carrier = seg.get("carrierCode") or seg.get("carrier")
                    if carrier and carrier not in carrier_codes:
                        carrier_codes.append(carrier)
                carriers = ",".join(carrier_codes) if carrier_codes else ""

                links = offer.get("links") or {}
                deeplink = None
                if isinstance(links, dict):
                    for v in links.values():
                        if isinstance(v, str) and v.startswith("http"):
                            deeplink = v
                            break

                opt = {
                    "index": i,
                    "type": "flight",
                    "from": o_code,
                    "to": d_code,
                    "depart": dep,
                    "arrive": arr,
                    "duration": duration,
                    "stops": stops,
                    "price": price_total,
                    "price_str": price_str,
                    "airline": airline,
                    "flight_no": flight_no,
                    "offer": offer,

                    # 🔴 THIS is the ONLY URL open option will use
                    "url": self._make_flight_booking_url({
                        "from": o_code,
                        "to": d_code,
                        "depart": dep,
                    }),
                }

                self.last_results.append(opt)

                rows.append([
                    i,
                    dep or "-",
                    arr or "-",
                    duration or "-",
                    price_str,
                    f"{stops} stop" if stops == 1 else f"{stops} stops",
                    carriers,
                ])
            except Exception as e:
                print("(booking) parsing offer error:", e)
                continue

        print(f"\n(booking) Flights {o_code} → {d_code} on {depart_iso}:")
        self._print_table(rows, ["#", "Depart", "Arrive", "Duration", "Price", "Stops", "Carriers"])
        speak(
            f"I've listed {len(rows)} flight options in the terminal. "
            f"You can say 'open option 1' to view it in browser, or 'book option 1' to book here."
        )

    def _print_table(self, rows, headers):
        if not rows:
            print("(booking) No rows to display.")
            return
        widths = [len(h) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(str(c)))
        def fmt(row):
            return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        print(fmt(headers))
        print("  ".join("-" * w for w in widths))
        for r in rows:
            print(fmt(r))
            
    def _open_option(self, index: int):
        if not self.last_results:
            speak("No options available.")
            return

        if index < 1 or index > len(self.last_results):
            speak("That option number is out of range.")
            return

        opt = self.last_results[index - 1]
        url = opt.get("url")

        if not url:
            speak("This option does not have a booking link.")
            return

        print(f"(booking) Opening option {index}: {url}")

        ok = open_with(
            ["firefox", "google-chrome", "chromium-browser", "xdg-open"],
            [url],
        )

        if ok:
            speak(f"Opening option {index} in your browser.")
        else:
            speak("Couldn't open the browser. Link printed in the terminal.")
            print(url)

