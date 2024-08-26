# Perfectly Plausible Captions
# Enriched Simplified Perfectly Plausible Captions

perfect_captions_short = [
    "A biker overlooks distant mountains.",
    "Motorcyclist admires mountain view.",
    "Rider pauses by mountain scenery.",
    "Biker gazes at misty peaks.",
    "Motorbike near a mountain vista.",
    "Rider enjoying mountain landscape.",
    "Biker at a scenic overlook.",
    "Motorcycle facing the mountains.",
    "Rider contemplates mountain range.",
    "Biker at mountain roadside.",
    "Motorcyclist views hilly terrain.",
    "Biker against mountain backdrop.",
    "Rider near mountain summit.",
    "Motorcycle by a mountain path.",
    "Biker watching over mountains.",
    "Rider reflects at mountain edge.",
    "Motorbike parked with mountain view.",
    "Biker enjoys a mountain moment.",
    "Rider at mountain viewpoint.",
    "Motorcycle rests by mountainside."
]
perfect_captions = [
    "A motorcyclist stops on a dirt road to take in the view of the distant mountains.",
    "A rider in a red jacket pauses to enjoy the mountain scenery.",
    "A biker atop a hill gazes at the mountain range ahead.",
    "A traveler on a motorcycle looks out at the foggy mountain peaks.",
    "A person on a motorbike takes a break to admire the mountainous landscape.",
    "A motorcycle enthusiast savors the view from a high vantage point.",
    "A lone motorcyclist contemplates the mountains under a cloudy sky.",
    "A biker leans on a guardrail, surrounded by a breathtaking mountain view.",
    "A motorcycle parked near a lookout point, with mountains in the backdrop.",
    "A rider takes a moment to absorb the tranquil mountain atmosphere.",
    "A red-clad motorcyclist enjoys the panoramic view of the alpine range.",
    "A biker's journey paused by the impressive sight of towering mountains.",
    "A motorcycle sits idle as the rider surveys the mountain vistas.",
    "A rider in reflective gear observes the mist-covered mountain tops.",
    "A motorbike journey brought to a contemplative halt by the mountain scenery.",
    "A biker's silhouette against the mountainous horizon during a rest stop.",
    "A rider examines the rocky mountain terrain from a roadside stop.",
    "A motorcycle rests at the edge of a mountain path, overlooking the valley.",
    "A road adventurer takes in the expansive view of the mountain ranges.",
    "A biker on a rural mountain road stops to appreciate the natural surroundings."
]

# Captions with Some Incorrect Descriptions
incorrect_captions = [
    "A cyclist pauses on a hill mistaking distant trees for mountains.",
    "A motorbike rider looks at a large hill incorrectly described as a mountain.",
    "A biker near the sea with the ocean view mistaken for mountain scenery.",
    "A person on a scooter on a city street with buildings perceived as mountains.",
    "A motorcyclist on a country road where distant farms are seen as mountain ranges.",
    "A rider at the edge of a forest with the trees described as a mountain vista.",
    "A biker on a bridge overlooking a river confused with a mountainous view.",
    "A cyclist in a park where the skyline is thought to be a mountain range.",
    "A person on a bike in a valley mistaking rolling hills for high mountains.",
    "A motorbike parked in a desert area with sand dunes perceived as mountains.",
    "A rider in an urban area looks at tall buildings envisioned as mountains.",
    "A biker on a coastal path with the cliffside seen as mountainous terrain.",
    "A motorcyclist in a meadow where the open field is seen as a mountain range.",
    "A person on a bike on a flat plain with distant clouds imagined as mountains.",
    "A rider on a rooftop mistaking the cityscape for a mountain horizon.",
    "A cyclist on a farm road where the barns and fields are likened to mountains.",
    "A biker beside a lake with the water's edge seen as the base of mountains.",
    "A motorcyclist on a suburban street with houses imagined as mountain scenery.",
    "A rider in a garden where the landscaping is mistaken for mountainous terrain.",
    "A biker in a snowy field where the snowdrifts are perceived as mountain ranges."
]


# Captions Irrelevant to the Image
irrelevant_captions = [
    "A chef preparing sushi in a busy Japanese restaurant.",
    "Children playing soccer on a sunny field in the park.",
    "A librarian organizing books in a large, quiet library.",
    "An artist painting a mural on an urban street wall.",
    "A farmer harvesting apples in a lush orchard.",
    "A scientist conducting experiments in a modern laboratory.",
    "A group of friends camping and sitting around a bonfire.",
    "A teacher giving a lecture in a crowded classroom.",
    "A barista crafting latte art in a cozy café.",
    "A florist arranging colorful flowers in a boutique.",
    "A pianist performing at a concert hall.",
    "A diver exploring a coral reef under the sea.",
    "A pilot flying a plane above the clouds.",
    "A baker kneading dough in a cozy, warm bakery kitchen.",
    "A baker making bread in a traditional stone oven.",
    "A dancer rehearsing in a bright, spacious studio.",
    "A carpenter building furniture in a woodworking shop.",
    "A fisherman casting a net on a serene lake.",
    "A painter working on a still life in an art studio.",
    "A jogger running through a city park at sunrise."
]

# Nonsensical, Broken Captions
nonsensical_captions = [
    "Blueberry sunshine, velvet skateboard.",
    "Rainbow spaghetti, moonlight tambourine.",
    "Glitter tornado, caramel trampoline.",
    "Marshmallow lantern, quantum pebble.",
    "Sunshine parrot, disco teapot.",
    "Whirlwind pancake, galaxy muffin.",
    "Starlight cupcake, echo jellybean.",
    "Polka-dot elephant, chocolate comet.",
    "Bubblegum spaceship, pineapple clock.",
    "Velvet giraffe, thunderstorm sandwich.",
    "Quantum flamingo, lollipop hurricane.",
    "Moonbeam accordion, confetti raccoon.",
    "Stardust violin, peacock sunrise.",
    "Marmalade rocket, tinsel kangaroo.",
    "Watermelon typewriter, frostbite carousel.",
    "Lemonade cactus, snowflake harmonica.",
    "Cupcake volcano, sequin astronaut.",
    "Glitter hurricane, cotton candy penguin.",
    "Neon zebra, butterscotch cloud.",
    "Taffy tornado, velvet mermaid, marshmallow eclipse."
]

# それぞれのカテゴリごとにリストをまとめる
# all_captions = {
#     "Lv.5: Perfectly Plausible Captions": perfect_captions,
#     "Lv.4: Perfectly Plausible Captions": perfect_captions_short,
#     "Lv.3: Captions with Some Incorrect Descriptions": incorrect_captions,
#     "Lv.2: Captions Irrelevant to the Image": irrelevant_captions,
#     "Lv.1: Nonsensical, Broken Captions": nonsensical_captions
# }
all_captions = {
    4: perfect_captions,
    3: perfect_captions_short,
    2: incorrect_captions,
    1: irrelevant_captions
}