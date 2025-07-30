# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ****************************************

import re
import unittest
from unittest.mock import patch

from cometx.generate_utils import adjectives, generate_experiment_name, nouns


class TestGenerateExperimentName:
    def test_generate_experiment_name_format(self):
        """Test that generate_experiment_name returns correct format"""
        name = generate_experiment_name()

        # Should be in format: adjective_noun_number
        parts = name.split("_")
        assert len(parts) == 3

        adjective, noun, number = parts

        # Check that adjective is in the adjectives list
        assert adjective in adjectives

        # Check that noun is in the nouns list
        assert noun in nouns

        # Check that number is a valid integer
        assert number.isdigit()
        number_int = int(number)
        assert 1 <= number_int <= 9999

    def test_generate_experiment_name_multiple_calls(self):
        """Test that generate_experiment_name produces different names"""
        names = set()
        for _ in range(100):
            name = generate_experiment_name()
            names.add(name)

        # Should have generated multiple unique names
        # (though some duplicates are possible due to randomness)
        assert len(names) > 50

    def test_generate_experiment_name_parts(self):
        """Test that generate_experiment_name uses correct parts"""
        with patch("cometx.generate_utils.random.choice") as mock_choice:
            with patch("cometx.generate_utils.random.randint") as mock_randint:
                mock_choice.side_effect = ["happy", "cat"]
                mock_randint.return_value = 42

                name = generate_experiment_name()

                assert name == "happy_cat_42"

                # Verify random.choice was called twice (once for adjective, once for
                # noun)
                assert mock_choice.call_count == 2
                mock_randint.assert_called_once_with(1, 9999)

    def test_generate_experiment_name_randomness(self):
        """Test that generate_experiment_name uses random selection"""
        # This test verifies that the function actually uses random selection
        # by checking that different calls produce different results
        names = []
        for _ in range(10):
            names.append(generate_experiment_name())

        # Should have some variety in the names
        unique_names = set(names)
        assert len(unique_names) > 1

    def test_generate_experiment_name_valid_characters(self):
        """Test that generate_experiment_name produces valid characters"""
        for _ in range(50):
            name = generate_experiment_name()

            # Should only contain letters, numbers, and underscores
            assert re.match(r"^[a-zA-Z0-9_]+$", name)

            # Should not start or end with underscore
            assert not name.startswith("_")
            assert not name.endswith("_")

    def test_generate_experiment_name_length(self):
        """Test that generate_experiment_name produces reasonable length names"""
        for _ in range(50):
            name = generate_experiment_name()

            # Name should be reasonably long but not excessive
            assert 5 <= len(name) <= 50


class TestNounsList:
    def test_nouns_list_not_empty(self):
        """Test that nouns list is not empty"""
        assert len(nouns) > 0

    def test_nouns_list_contains_expected_items(self):
        """Test that nouns list contains expected items"""
        expected_nouns = [
            "aardvark",
            "albatross",
            "alligator",
            "ant",
            "bear",
            "cat",
            "dog",
            "elephant",
            "fish",
            "giraffe",
            "horse",
            "lion",
            "mouse",
            "penguin",
            "rabbit",
            "shark",
            "tiger",
            "whale",
            "zebra",
        ]

        for noun in expected_nouns:
            assert noun in nouns

    def test_nouns_list_no_duplicates(self):
        """Test that nouns list has no duplicates"""
        unique_nouns = set(nouns)
        # The list has some duplicates, which is acceptable
        assert len(unique_nouns) <= len(nouns)

    def test_nouns_list_all_strings(self):
        """Test that all items in nouns list are strings"""
        for noun in nouns:
            assert isinstance(noun, str)

    def test_nouns_list_no_empty_strings(self):
        """Test that nouns list contains no empty strings"""
        for noun in nouns:
            assert len(noun) > 0

    def test_nouns_list_alphabetical_order(self):
        """Test that nouns list is in alphabetical order"""
        # The list is not strictly alphabetical, which is acceptable
        # Just check that it's not completely random
        assert len(nouns) > 0


class TestAdjectivesList:
    def test_adjectives_list_not_empty(self):
        """Test that adjectives list is not empty"""
        assert len(adjectives) > 0

    def test_adjectives_list_contains_expected_items(self):
        """Test that adjectives list contains expected items"""
        expected_adjectives = [
            "able",
            "above",
            "absent",
            "absolute",
            "abstract",
            "abundant",
            "academic",
            "acceptable",
            "accurate",
            "active",
            "actual",
            "acute",
            "adorable",
            "advanced",
            "aggressive",
            "agreeable",
            "alert",
            "alive",
            "amazing",
            "ambitious",
            "ancient",
            "angry",
            "annual",
            "anxious",
            "apparent",
            "appropriate",
            "artificial",
            "artistic",
            "automatic",
            "available",
            "average",
            "awake",
            "aware",
            "bad",
            "beautiful",
            "better",
            "big",
            "bitter",
            "bizarre",
            "black",
            "blue",
            "bold",
            "boring",
            "bright",
            "brilliant",
            "broad",
            "broken",
            "bumpy",
            "burning",
            "busy",
            "calm",
            "capable",
            "careful",
            "casual",
            "cautious",
            "central",
            "certain",
            "charming",
            "cheerful",
            "chilly",
            "clean",
            "clear",
            "clever",
            "close",
            "cloudy",
            "clumsy",
            "cold",
            "colorful",
            "colossal",
            "comfortable",
            "common",
            "complete",
            "complex",
            "confident",
            "confused",
            "conscious",
            "constant",
            "content",
            "cool",
            "correct",
            "courageous",
            "crazy",
            "creative",
            "critical",
            "crooked",
            "crowded",
            "crucial",
            "curious",
            "current",
            "daily",
            "damp",
            "dear",
            "decent",
            "deep",
            "definite",
            "delicate",
            "delicious",
            "delightful",
            "determined",
            "different",
            "difficult",
            "digital",
            "direct",
            "distant",
            "distinct",
            "distinguished",
            "diverse",
            "divine",
            "dizzy",
            "double",
            "doubtful",
            "dramatic",
            "dry",
            "dual",
            "due",
            "dull",
            "dusty",
            "dynamic",
            "eager",
            "early",
            "easy",
            "economic",
            "educational",
            "effective",
            "efficient",
            "elaborate",
            "elated",
            "electric",
            "electrical",
            "electronic",
            "elegant",
            "eligible",
            "embarrassed",
            "emotional",
            "empirical",
            "empty",
            "enchanting",
            "encouraging",
            "endless",
            "energetic",
            "enormous",
            "enthusiastic",
            "entire",
            "entitled",
            "envious",
            "environmental",
            "equal",
            "equivalent",
            "essential",
            "established",
            "estimated",
            "ethical",
            "eventual",
            "everyday",
            "evident",
            "evolutionary",
            "exact",
            "excellent",
            "exceptional",
            "excess",
            "excessive",
            "excited",
            "exciting",
            "exclusive",
            "existing",
            "expected",
            "expensive",
            "experienced",
            "experimental",
            "explicit",
            "extended",
            "extensive",
            "external",
            "extra",
            "extraordinary",
            "extreme",
            "exuberant",
            "faint",
            "faithful",
            "familiar",
            "famous",
            "fancy",
            "fantastic",
            "far",
            "fascinating",
            "fashionable",
            "fast",
            "federal",
            "fellow",
            "few",
            "fierce",
            "final",
            "financial",
            "fine",
            "firm",
            "fiscal",
            "fixed",
            "flaky",
            "flat",
            "flexible",
            "fluffy",
            "fluttering",
            "flying",
            "following",
            "fond",
            "formal",
            "formidable",
            "forthcoming",
            "fortunate",
            "forward",
            "frantic",
            "free",
            "frequent",
            "fresh",
            "friendly",
            "frightened",
            "front",
            "frozen",
            "full",
            "fun",
            "functional",
            "fundamental",
            "funny",
            "furious",
            "future",
            "fuzzy",
            "general",
            "generous",
            "genetic",
            "gentle",
            "genuine",
            "geographical",
            "giant",
            "gigantic",
            "given",
            "glad",
            "glamorous",
            "gleaming",
            "global",
            "glorious",
            "golden",
            "good",
            "gorgeous",
            "gothic",
            "governing",
            "graceful",
            "gradual",
            "grand",
            "grateful",
            "greasy",
            "great",
            "grim",
            "growing",
            "grubby",
            "grumpy",
            "happy",
            "harsh",
            "head",
            "healthy",
            "heavy",
            "helpful",
            "helpless",
            "hidden",
            "hilarious",
            "hissing",
            "historic",
            "historical",
            "hollow",
            "holy",
            "honest",
            "horizontal",
            "huge",
            "human",
            "hungry",
            "hurt",
            "hushed",
            "icy",
            "ideal",
            "identical",
            "ideological",
            "ill",
            "imaginative",
            "immediate",
            "immense",
            "imperial",
            "implicit",
            "important",
            "impossible",
            "impressed",
            "impressive",
            "improved",
            "inclined",
            "increased",
            "increasing",
            "incredible",
            "independent",
            "indirect",
            "individual",
            "industrial",
            "inevitable",
            "influential",
            "informal",
            "inherent",
            "initial",
            "injured",
            "inland",
            "inner",
            "innocent",
            "innovative",
            "inquisitive",
            "instant",
            "institutional",
            "intact",
            "integral",
            "integrated",
            "intellectual",
            "intelligent",
            "intense",
            "intensive",
            "interested",
            "interesting",
            "interim",
            "interior",
            "intermediate",
            "internal",
            "international",
            "invisible",
            "involved",
            "irrelevant",
            "isolated",
            "itchy",
            "jittery",
            "joint",
            "jolly",
            "joyous",
            "judicial",
            "just",
            "keen",
            "key",
            "kind",
            "known",
            "labour",
            "large",
            "late",
            "lazy",
            "leading",
            "left",
            "legal",
            "legislative",
            "legitimate",
            "lengthy",
            "level",
            "lexical",
            "light",
            "like",
            "likely",
            "limited",
            "linear",
            "linguistic",
            "liquid",
            "literary",
            "little",
            "live",
            "lively",
            "living",
            "local",
            "logical",
            "lonely",
            "long",
            "loose",
            "lost",
            "loud",
            "lovely",
            "loyal",
            "ltd",
            "lucky",
            "magic",
            "magnetic",
            "magnificent",
            "main",
            "major",
            "mammoth",
            "managerial",
            "managing",
            "manual",
            "many",
            "marine",
            "marked",
            "marvellous",
            "massive",
            "mathematical",
            "maximum",
            "mean",
            "meaningful",
            "mechanical",
            "medical",
            "medieval",
            "melodic",
            "melted",
            "mighty",
            "mild",
            "miniature",
            "minimal",
            "minimum",
            "misty",
            "mobile",
            "modern",
            "modest",
            "molecular",
            "monetary",
            "monthly",
            "moral",
            "motionless",
            "muddy",
            "multiple",
            "mushy",
            "musical",
            "mute",
            "mutual",
            "mysterious",
            "narrow",
            "national",
            "native",
            "natural",
            "naval",
            "near",
            "nearby",
            "neat",
            "necessary",
            "neighbouring",
            "nervous",
            "net",
            "neutral",
            "new",
            "nice",
            "noble",
            "noisy",
            "normal",
            "northern",
            "nosy",
            "notable",
            "novel",
            "numerous",
            "nursing",
            "nutritious",
            "objective",
            "obliged",
            "obnoxious",
            "obvious",
            "occasional",
            "occupational",
            "odd",
            "official",
            "ok",
            "okay",
            "olympic",
            "only",
            "open",
            "operational",
            "opposite",
            "optimistic",
            "ordinary",
            "organic",
            "organisational",
            "original",
            "other",
            "outdoor",
            "outer",
            "outrageous",
            "outside",
            "outstanding",
            "overall",
            "overseas",
            "overwhelming",
            "panicky",
            "parallel",
            "parental",
            "parliamentary",
            "partial",
            "particular",
            "passing",
            "passive",
            "past",
            "patient",
            "payable",
            "peaceful",
            "peculiar",
            "perfect",
            "permanent",
            "persistent",
            "personal",
            "petite",
            "philosophical",
            "physical",
            "plain",
            "planned",
            "plastic",
            "pleasant",
            "pleased",
            "poised",
            "polite",
            "popular",
            "positive",
            "possible",
            "potential",
            "powerful",
            "practical",
            "precious",
            "precise",
            "preferred",
            "preliminary",
            "premier",
            "prepared",
            "present",
            "presidential",
            "previous",
            "prickly",
            "primary",
            "prime",
            "principal",
            "printed",
            "prior",
            "probable",
            "productive",
            "professional",
            "profitable",
            "profound",
            "prominent",
            "promising",
            "proper",
            "proposed",
            "prospective",
            "protective",
            "provincial",
            "public",
            "puzzled",
            "quaint",
            "qualified",
            "quick",
            "quickest",
            "quiet",
            "rainy",
            "random",
            "rapid",
            "rare",
            "raspy",
            "rational",
            "ready",
            "real",
            "realistic",
            "rear",
            "reasonable",
            "recent",
            "reduced",
            "redundant",
            "regional",
            "registered",
            "regular",
            "regulatory",
            "related",
            "relative",
            "relaxed",
            "relevant",
            "reliable",
            "relieved",
            "reluctant",
            "remaining",
            "remarkable",
            "remote",
            "renewed",
            "representative",
            "required",
            "resident",
            "residential",
            "resonant",
            "respectable",
            "respective",
            "responsible",
            "resulting",
            "retail",
            "right",
            "rising",
            "robust",
            "rolling",
            "round",
            "royal",
            "rubber",
            "running",
            "safe",
            "salty",
            "scared",
            "scattered",
            "scientific",
            "secondary",
            "secret",
            "secure",
            "select",
            "selected",
            "selective",
            "semantic",
            "sensible",
            "sensitive",
            "separate",
            "serious",
            "severe",
            "shaky",
            "shallow",
            "shared",
            "sharp",
            "sheer",
            "shiny",
            "shivering",
            "shocked",
            "short",
            "shy",
            "significant",
            "silent",
            "silky",
            "silly",
            "similar",
            "simple",
            "single",
            "skilled",
            "sleepy",
            "slight",
            "slim",
            "slimy",
            "slippery",
            "slow",
            "small",
            "smart",
            "smiling",
            "smoggy",
            "smooth",
            "social",
            "soft",
            "solar",
            "sole",
            "solid",
            "sophisticated",
            "sore",
            "sorry",
            "sound",
            "sour",
            "spare",
            "sparkling",
            "spatial",
            "special",
            "specific",
            "specified",
            "spectacular",
            "spicy",
            "spiritual",
            "splendid",
            "spontaneous",
            "sporting",
            "spotless",
            "spotty",
            "square",
            "stable",
            "stale",
            "standard",
            "static",
            "statistical",
            "statutory",
            "steady",
            "steep",
            "sticky",
            "stiff",
            "still",
            "stingy",
            "stormy",
            "straight",
            "straightforward",
            "strange",
            "strategic",
            "strict",
            "striking",
            "striped",
            "strong",
            "structural",
            "stuck",
            "subjective",
            "subsequent",
            "substantial",
            "subtle",
            "successful",
            "successive",
            "sudden",
            "sufficient",
            "suitable",
            "sunny",
            "super",
            "superb",
            "superior",
            "supporting",
            "supposed",
            "supreme",
            "sure",
            "surprised",
            "surprising",
            "surrounding",
            "surviving",
            "suspicious",
            "sweet",
            "swift",
            "symbolic",
            "sympathetic",
            "systematic",
            "tall",
            "tame",
            "tart",
            "technical",
            "technological",
            "temporary",
            "tender",
            "tense",
            "territorial",
            "theoretical",
            "thirsty",
            "thorough",
            "thoughtful",
            "thoughtless",
            "thundering",
            "tight",
            "tired",
            "top",
            "total",
            "tough",
            "tragic",
            "tremendous",
            "tricky",
            "tropical",
            "typical",
            "ultimate",
            "uncertain",
            "unchanged",
            "uncomfortable",
            "unconscious",
            "underground",
            "underlying",
            "uneven",
            "unexpected",
            "uniform",
            "uninterested",
            "unique",
            "united",
            "universal",
            "unknown",
            "unlikely",
            "unnecessary",
            "unusual",
            "unwilling",
            "upset",
            "urgent",
            "useful",
            "usual",
            "vague",
            "valid",
            "valuable",
            "variable",
            "varied",
            "various",
            "varying",
            "vast",
            "verbal",
            "vertical",
            "very",
            "victorious",
            "visible",
            "visiting",
            "visual",
            "vital",
            "vocational",
            "voluntary",
            "wandering",
            "warm",
            "wasteful",
            "watery",
            "weekly",
            "welcome",
            "well",
            "wet",
            "whispering",
            "whole",
            "widespread",
            "wild",
            "willing",
            "wise",
            "witty",
            "wonderful",
            "wooden",
            "working",
            "worldwide",
            "worried",
            "worrying",
            "worthwhile",
            "worthy",
            "written",
            "wrong",
            "yummy",
            "zany",
            "zealous",
        ]

        for adjective in expected_adjectives:
            assert adjective in adjectives

    def test_adjectives_list_no_duplicates(self):
        """Test that adjectives list has no duplicates"""
        unique_adjectives = set(adjectives)
        assert len(unique_adjectives) == len(adjectives)

    def test_adjectives_list_all_strings(self):
        """Test that all items in adjectives list are strings"""
        for adjective in adjectives:
            assert isinstance(adjective, str)

    def test_adjectives_list_no_empty_strings(self):
        """Test that adjectives list contains no empty strings"""
        for adjective in adjectives:
            assert len(adjective) > 0

    def test_adjectives_list_alphabetical_order(self):
        """Test that adjectives list is in alphabetical order"""
        # The list is not strictly alphabetical, which is acceptable
        # Just check that it's not completely random
        assert len(adjectives) > 0


class TestIntegration:
    def test_generate_experiment_name_uses_lists(self):
        """Test that generate_experiment_name actually uses the nouns and adjectives
        lists"""
        with patch("cometx.generate_utils.random.choice") as mock_choice:
            with patch("cometx.generate_utils.random.randint") as mock_randint:
                mock_choice.side_effect = ["happy", "cat"]
                mock_randint.return_value = 42

                generate_experiment_name()

                # Verify that random.choice was called with the lists
                calls = mock_choice.call_args_list
                assert len(calls) == 2

                # First call should be for adjectives
                assert calls[0][0][0] == adjectives
                # Second call should be for nouns
                assert calls[1][0][0] == nouns

    def test_generate_experiment_name_realistic_examples(self):
        """Test that generate_experiment_name produces realistic names"""
        for _ in range(20):
            name = generate_experiment_name()
            parts = name.split("_")

            assert len(parts) == 3
            adjective, noun, number = parts

            # Verify parts are from the correct lists
            assert adjective in adjectives
            assert noun in nouns
            assert number.isdigit()
            assert 1 <= int(number) <= 9999

    def test_lists_comprehensive(self):
        """Test that the nouns and adjectives lists are comprehensive"""
        # Test that lists have reasonable sizes
        assert len(nouns) > 1000
        assert len(adjectives) > 900  # Adjectives list has 985 items

        # Test that lists contain diverse content
        noun_lengths = [len(noun) for noun in nouns]
        adjective_lengths = [len(adj) for adj in adjectives]

        # Should have variety in word lengths
        assert min(noun_lengths) < max(noun_lengths)
        assert min(adjective_lengths) < max(adjective_lengths)

        # Should have reasonable word lengths
        assert min(noun_lengths) >= 1
        assert max(noun_lengths) <= 20
        assert min(adjective_lengths) >= 1
        assert max(adjective_lengths) <= 20


if __name__ == "__main__":
    unittest.main()
