import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./qwen_finetuned"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load Model (in 4-bit to save memory)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 3. Prepare Input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze the 5G wireless network drive-test user plane data and engineering parameters.\nIdentify the reason for the throughput dropping below 600Mbps in certain road sections.\nFrom the following 8 potential root causes, select the most likely one and enclose its number in \\boxed{{}} in the final answer.\n\nC1: The serving cell's downtilt angle is too large, causing weak coverage at the far end.\nC2: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.\nC3: A neighboring cell provides higher throughput.\nC4: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.\nC5: Frequent handovers degrade performance.\nC6: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.\nC7: Test vehicle speed exceeds 40km/h, impacting user throughput.\nC8: Average scheduled RBs are below 160, affecting throughput.\n\nGiven:\n- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees. Other values represent the actual downtilt angle in degrees.\n\nBeam Scenario and Vertical Beamwidth Relationships:\n- When the cell's Beam Scenario is set to Default or SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees.\n- When the cell's Beam Scenario is set to SCENARIO_6 to SCENARIO_11, the vertical beamwidth is 12 degrees.\n- When the cell's Beam Scenario is set to SCENARIO_12 or above, the vertical beamwidth is 25 degrees.\n\nUser plane drive test data as follows\uff1a\n\nTimestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|5G KPI PCell Layer2 MAC DL Throughput [Mbps]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 Filtered Tx BRSRP [dBm]|5G KPI PCell Layer1 DL RB Num (Including 0)\n2025-05-07 15:11:42.000000|128.187027|32.582075|23|217|-83.75|13.15|1200.17|96|185|71|106|498|-101.79|-106.06|-112.11|-119.76|-121.04|210.67\n2025-05-07 15:11:43.000000|128.187041|32.582075|20|217|-80.35|13.02|1183.75|96|185|498|660|265|-100.3|-107.12|-113.52|-118.97|-124.2|210.86\n2025-05-07 15:11:44.000000|128.187049|32.582086|18|217|-84.21|11.15|959.79|96|185|498|265|660|-96.66|-106.59|-112.56|-119.2|-127.46|211.18\n2025-05-07 15:11:45.000000|128.187056|32.582098|21|217|-84.65|10.93|951.47|96|185|-|-|-|-94.06|-109.01|-|-|-|210.71\n2025-05-07 15:11:46.000000|128.187063|32.582117|15|217|-84.85|8.23|417.25|96|185|71|660|-|-91.96|-107.04|-110.67|-117.84|-|104.9\n2025-05-07 15:11:47.000000|128.187085|32.582125|1|217|-86.51|6.94|400.74|96|185|-|-|-|-89.88|-101.89|-|-|-|105.13\n2025-05-07 15:11:48.000000|128.187093|32.582136|34|96|-89.35|9.42|466.05|217|185|265|-|-|-93.07|-103.99|-110.11|-|-|105.195\n2025-05-07 15:11:49.000000|128.187093|32.582136|10|96|-84.33|11.95|423.05|217|185|-|-|-|-94.73|-98.2|-|-|-|105.345\n2025-05-07 15:11:50.000000|128.187093|32.582136|23|96|-83.09|9.39|884.29|217|185|-|-|-|-92.16|-103.03|-|-|-|210.0\n2025-05-07 15:11:51.000000|128.187093|32.582136|14|96|-86.55|7.75|993.62|217|185|-|-|-|-90.79|-102.97|-|-|-|210.15\n\n\nEngeneering parameters data as follows\uff1a\n\ngNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Beam Scenario|Height|PCI|TxRx Mode|Max Transmit Power|Antenna Model\n0033978|12|128.195732|32.588859|55|9|255|0|DEFAULT|39.2|435|32T32R|34.9|NR AAU 3\n0034036|14|128.200589|32.594674|0|0|255|0|EXPAND_SCENARIO_2|0.0|868|4T4R|46.0|Other\n0033166|25|128.189449|32.574162|330|17|9|-5|SCENARIO_2|31.1|265|64T64R|34.9|NR AAU 2\n0033164|27|128.189091|32.580931|240|3|10|0|DEFAULT|29.7|71|32T32R|34.9|NR AAU 1\n0033162|2|128.186111|32.566464|300|8|12|0|DEFAULT|34.0|106|64T64R|34.9|NR AAU 2\n0034039|3|128.191866|32.578753|290|39|255|0|DEFAULT|123.0|498|64T64R|34.9|NR AAU 2\n0000293|0|128.175743|32.589441|192|9|7|0|SCENARIO_9|3.0|660|64T64R|34.9|NR AAU 2\n0033317|14|128.185221|32.579659|55|10|11|0|DEFAULT|25.9|96|64T64R|34.9|NR AAU 2\n0033317|13|128.186916|32.580641|0|5|0|0|DEFAULT|17.4|217|32T32R|34.9|NR AAU 1\n0033164|29|128.185307|32.581601|195|8|12|0|DEFAULT|31.6|185|32T32R|34.9|NR AAU 1\n"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. Generate Response
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\nResponse:\n", response)
