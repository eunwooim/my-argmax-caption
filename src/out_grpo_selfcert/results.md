# Step 2: GRPO with selfcert reward

Reward: **self-certainty(policy)** (frozen). Policy: Qwen/Qwen2.5-VL-7B-Instruct + LoRA r=16. Caption prompt: `Describe this image.`, up to 128 new tokens. Rollouts 4, batch 8 images, 300 steps, lr 1e-05, KL beta 0.04, loss norm const.

Before = BeforeRlModelCaption (base model). After = Argmax* (GRPO-trained). Greedy decoding for both. Intervals: 95% bootstrap over images.

`reward` is the score that was optimised, so it should go up by construction. The other rows were never trained on: if they do not move with it, the gain is specific to the reward (reward hacking), not better captions.

| set | scorer | before | after | after - before | after wins |
|---|---|---|---|---|---|
| coco_val (n=500) | reward (self-certainty(policy)) | 13.1207 | 14.6961 | +1.5754 [+1.5140, +1.6372] | 99.6% [99.0, 100.0] |
| coco_val (n=500) | self (S2-centered@L24) | 0.6964 | 0.6614 | -0.0350 [-0.0417, -0.0283] | 32.2% [28.2, 36.2] |
| coco_val (n=500) | S1 (S1-centered@L28) | 0.7870 | 0.7598 | -0.0271 [-0.0365, -0.0180] | 41.4% [37.0, 45.8] |
| coco_val (n=500) | S3 (S3-centered@L28) | 0.4332 | 0.3725 | -0.0608 [-0.0694, -0.0520] | 28.8% [24.8, 32.8] |
| coco_val (n=500) | S0 (S0-centered@L28) | 0.2012 | 0.1432 | -0.0580 [-0.0696, -0.0465] | 33.0% [29.0, 37.2] |
| docci_ood (n=200) | reward (self-certainty(policy)) | 15.0401 | 16.6303 | +1.5902 [+1.4731, +1.7088] | 99.0% [97.5, 100.0] |
| docci_ood (n=200) | self (S2-centered@L24) | 0.6701 | 0.6494 | -0.0207 [-0.0322, -0.0096] | 40.0% [33.0, 47.0] |
| docci_ood (n=200) | S1 (S1-centered@L28) | 0.7687 | 0.7253 | -0.0434 [-0.0560, -0.0313] | 32.0% [25.5, 38.5] |
| docci_ood (n=200) | S3 (S3-centered@L28) | 0.4265 | 0.4079 | -0.0186 [-0.0304, -0.0067] | 50.0% [43.0, 57.0] |
| docci_ood (n=200) | S0 (S0-centered@L28) | 0.0723 | 0.0611 | -0.0112 [-0.0220, -0.0004] | 44.0% [37.0, 50.5] |

| set | words before | words after |
|---|---|---|
| coco_val | 90.1 | 84.9 |
| docci_ood | 92.4 | 77.9 |

Step 1 found this score does not rise with more true detail, so shrinking captions are a warning sign.

## Examples (coco_val)

- **['coco', 365098]** before (12.726): The image shows a close-up of a small, dark-colored boat floating on a body of water. The boat appears to be made of a material like fiberglass or plastic, and it has a sleek, elongated design. The water is a vibrant turquoise color, suggesting a calm and clear day. The boat has a visible registration number, "M 7954," written on its side. The edge of the boat is slightly worn, and there are some green leaves in the foreground, possibly from a nearby tree, adding a natural element to the scene. The overall atmosphere of the image is serene and peaceful.
  after (14.681): The image shows a close-up view of a small boat floating on water. Here are some details:

1. **Boat**: The boat appears to be made of metal or a similar material. It has a sleek, elongated shape typical of small rowboats or canoes.
2. **Water**: The boat is floating on a body of water that has a bluish-green hue, suggesting it might be a lake or river.
3. **Details on the Boat**: There is some text or markings visible on the side of the boat, but it is not entirely clear what it says.
4. **Background**: The background is slightly
- **['coco', 398905]** before (13.019): The image shows a person performing on stage, playing a Roland keyboard. The individual has curly hair and a beard, and is wearing a denim jacket over a light-colored shirt. They are positioned in front of a microphone, suggesting they are singing or speaking into it. The background features a large, circular, colorful projection, adding a vibrant and dynamic element to the scene. The setting appears to be a live performance or concert.
  after (14.220): The image shows a person playing a Roland keyboard. Here are the details:

1. **Person**: The individual has curly hair and a beard. They are wearing a denim jacket over a light-colored shirt.
2. **Keyboard**: A Roland keyboard is in front of the person, indicating they are playing music.
3. **Microphone**: There is a microphone positioned in front of the keyboard, suggesting that the person might be singing or speaking while playing.
4. **Background**: The background appears to be a projection screen with abstract visuals, adding to the ambiance of the setting.

The overall scene suggests that this is a performance or a live
- **['coco', 35682]** before (12.780): The image shows a person sitting at a table in a café or bakery, holding a chocolate-covered donut. The table is set with various donuts, including one with chocolate glaze and sprinkles, another with a chocolate glaze and a cherry on top, and a third with a chocolate glaze and a decorative pattern. There is also a cup of coffee or a similar beverage with a latte art design. The person is wearing glasses and a plaid shirt. In the background, there are other patrons and a counter with shelves and displays. The setting appears to be a cozy and casual environment.
  after (15.292): The image shows a person sitting at a table in what appears to be a café or bakery. Here are the details:

1. **Person**: The individual in the image is holding a donut and appears to be posing for the photo.
2. **Food**: On the table in front of the person, there are several donuts of different types and colors. One donut is covered in chocolate and sprinkles, while others are plain or have chocolate glaze.
3. **Drinks**: There is a cup of what looks like coffee or hot chocolate on the table.
4. **Background**: In the background, there are other
- **['coco', 243867]** before (12.653): The image shows a white bus parked on the side of a street, with a silver car partially visible in the foreground on the left. The bus has a number "3804" displayed on its front. The scene appears to be in an urban area, with buildings and storefronts in the background. The sunlight is bright, creating lens flare effects in the image. The bus is parked near a curb, and there are yellow lines on the road indicating a no-parking zone. The overall setting suggests a typical city street during the daytime.
  after (14.215): The image shows the following elements:

1. **A white bus**: The bus is parked on the side of the road. It has the number "3804" displayed on its front. The bus appears to be a tour or public transportation vehicle.

2. **Cars**: There are two cars visible in the image. One is on the left side of the bus, and the other is partially visible on the right side.

3. **Buildings**: In the background, there are buildings that appear to be part of a commercial area. Some signage is visible on the buildings, but the text is not entirely clear.

4.
- **['coco', 483050]** before (13.573): The image shows a neatly made bed in a room with a warm, inviting ambiance. The bed has a wooden headboard and is dressed with white linens, including a white duvet and several white pillows. A single red accent pillow is placed in the center of the bed. The room's walls are painted a vibrant red, complementing the red accents on the bed. On either side of the bed are nightstands, each equipped with a lamp, a clock, and a telephone. The overall setting suggests a comfortable and stylish hotel room or a well-decorated bedroom.
  after (15.882): The image shows a neatly made bed in a room. Here are the details:

1. **Bed**: 
   - The bed has a wooden headboard.
   - The bedding consists of white sheets and multiple pillows. 
   - There is one red pillow placed in the center of the bed.

2. **Room Details**:
   - The walls are painted in a vibrant red color.
   - On either side of the bed, there are nightstands.
   - On the left nightstand, there is a lamp and a digital alarm clock.
   - On the right nightstand, there is another lamp and what appears to be

## Examples (docci_ood)

- **['docci', 'test_03155']** before (15.967): The image shows a blue BMW car parked on the side of a road. The car appears to be a BMW 4 Series Coupe, identifiable by its sleek design and the distinctive BMW logo on the wheels. The car is parked in a lot with a chain-link fence in the background, and there is a tennis court visible beyond the fence. The car's body is a vibrant blue color, and it has a modern, sporty look with alloy wheels. The setting appears to be an outdoor area, possibly near a school or community center, given the presence of the tennis court and the building in the background.
  after (18.227): The image shows a blue BMW car parked on the side of a road. The car appears to be a coupe model, characterized by its sleek and aerodynamic design. The wheels of the car are black with a multi-spoke design, and the overall condition of the car looks well-maintained. In the background, there is a chain-link fence, a tennis court, and some buildings, indicating that the car is parked in an outdoor area, possibly near a school or recreational facility.
- **['docci', 'test_03445']** before (15.357): The image depicts a highly detailed and artistic motorcycle sculpture displayed in what appears to be a gallery or museum setting. The motorcycle is designed with a fantastical and imaginative theme, featuring large, intricate wings that resemble those of a dragonfly or butterfly. The body of the motorcycle is adorned with various textures and colors, giving it a unique and almost otherworldly appearance. The wheels are also intricately designed, with a detailed spoke pattern that adds to the overall aesthetic. The background includes framed artwork and other sculptures, contributing to the artistic ambiance of the space. The motorcycle is mounted on a platform, and the surrounding area is well-lit,
  after (16.792): The image depicts a highly detailed and artistic motorcycle sculpture. Here are some key features of the image:

1. **Motorcycle Design**: The motorcycle has an unconventional and imaginative design. It features large, intricate wings extending from the front, giving it a dragonfly-like appearance. The body of the motorcycle is adorned with various textures and colors, including metallic and iridescent finishes.

2. **Wheels**: The wheels of the motorcycle are also highly detailed, with complex spoke patterns and a metallic sheen.

3. **Surroundings**: The motorcycle is displayed in what appears to be an art gallery or museum setting. In the background
- **['docci', 'test_00331']** before (14.862): The image shows a small, gray and white kitten peeking out from under a cardboard box. The kitten has a curious expression, with its eyes wide open and its head slightly tilted. The box has a yellow label with some text on it, though the text is not fully legible in the image. The background appears to be a plain, light-colored surface, possibly a floor or a table. The overall scene gives a sense of the kitten's playful and inquisitive nature.
  after (15.608): The image shows a small gray and white kitten peeking out from under a piece of furniture. The kitten has large, round eyes and appears to be looking directly at the camera. In the background, there is a cardboard box with a yellow label on it. The overall setting seems to be indoors, possibly in a living room or similar space.
- **['docci', 'test_02121']** before (14.026): The image depicts the interior of a cave or underground passage. The walls and ceiling are composed of rough, rocky surfaces with a mix of light and dark tones, suggesting a natural formation. There is a noticeable shadow on the wall, which appears to be cast by a person standing in the foreground, though the person is not visible in the image. The ground is uneven and appears to be covered with dirt and small rocks. In the background, there is a faint light source, possibly from a flashlight or headlamp, illuminating another person who is partially visible. The overall atmosphere is dim and mysterious, typical of an underground or cave environment.
  after (16.332): The image depicts the interior of a cave. Here are some details:

1. **Rock Formations**: The cave walls and ceiling are composed of various rock formations. The rocks have a rough texture and are primarily in shades of brown and gray.

2. **Shadow of a Person**: There is a shadow of a person standing in the cave. The shadow is cast on the wall, indicating that there is a light source coming from the right side of the image.

3. **Ground**: The ground inside the cave appears to be uneven and covered with small rocks and debris.

4. **Background**: In the background, there are more rock
- **['docci', 'test_04188']** before (15.102): The image shows two decks of Maverick playing cards placed on a woven, light brown mat. The cards are positioned on either side of a set of wooden dominoes. The dominoes are dark brown with white dots, and they are arranged in a scattered manner. The Maverick playing cards are in their original packaging, which features a red and white design with a spade symbol and the text "MAVERICK" prominently displayed. The packaging also includes the words "PLAYING CARDS" and "CARTES À JOUER" in both English and French. The overall setting appears to be a casual, indoor environment
  after (18.013): The image shows two decks of playing cards and a set of wooden dominoes. Here are the details:

1. **Playing Cards**: 
   - There are two decks of playing cards on either side of the image.
   - The cards are labeled "MAVERICK" and "PLAYING CARDS" in both English and French ("CARTES À JOUER").
   - The design on the cards features a spade symbol in the center.

2. **Wooden Dominoes**:
   - In the center of the image, there is a set of wooden dominoes.
   - The dominoes are

