#  

Using a set of 7K reviews with ratings of 1, and a set of 7K with ratings of 5, I found words that are unique to each set, then I kept the unique and removed the common words. Using the LDA approach (from the previous week), I extracted five groups of topics per each set of data.


```
Topics for reviews with ratings of 1

# Topic0 is probably on defective product / not as advertised
# leaked deflated listened seal advertisement scam intermittently defected restocking punching apparent resemble blinking reputable composite rude lighted caved fired soldered swelled plywood mushroom proprietary toaster hinge prolonged deceived conscience proceeded loyal becasue communicate clicked magnifying overpowering snail bogus mistaken barking cane darkest viewfinder pray pilled pint struggled reproductive appropriately balcony

# Topic1 is probably on cheap quality
# refund hands cheep wrinkled irritated wast repaired retrieve manually lightsaber hunk actuality stating processed chalkboard hatched gut frizzy recycling reflective trailer multicolored reordered oozed wobbly heirloom rigged spacer slimy ripoff regretted greet runny rolles gaping conform immediatly restick resembling smeared inevitably perfume inspected sheared squeezing tryed plasma flawed hopeful meat

# Topic2 is harder to interpret
# dissatisfied contacting pass advertising warped expired isnt advertise rusted dissapointment bummer inconsistent marginally thinner exchanged poured sorely crumbled remain scuff embarassed aimed integrity fraction reeked improperly rendering beard repairing aggravation assumed stained demanding poking shoved unknown flatten stamped shorted excuse thrift sanded hype tipping misuse guessed resetting misspelled tempting dissappointing

# Topic3 is harder to interpret
# incomplete assumed upsetting ripping counterfeit deceptive trusted adhere pumped polish unsatisfied postage disassemble cheapo unsafe disconnected corroded partly tasted mashed trashed landfill mailing rethink locked gnome disappearing contrary disposable circular dated convince tilting exposure functioned paperwork tiniest sway delux ripe chassis sole pices homely repurchased fasten verify extraordinary slushee stunned

# Topic4 is s harder to interpret
# slightest melted according reset unusable bummed dissappointed embarrassing dissapointing reception inspection realizing rubbed crank severely loading patiently reassemble superglue wiring pissed confirmation decorative slipped fails kinked ripping smooshed lowered advertises wishing inspected ballerina cheated health worthwhile pitched apology flush cage filed goop werent undersized annoyingly splintered unwearable assorted topple dispense
```

Similarly, for the reviews with ratings of 5, I exracted 5 topics as well. 

Next, I compared the resulting lists of words with the dictionary of word2vec ensuring the list only contained known words. I selected  300-dimensional embeddings.
Next, the embeddings within each topic are averaged, thus resulting in a vector of dimension 5 x 300. The vector was further reduced using the TSNE and plotted in 3D. 


The figures show somewhat clear separation of the five topics within each set.

![Figure_2--11](https://user-images.githubusercontent.com/20401990/156698828-6c3ade40-2e65-4a84-8f16-4d465755681d.png)
![Figure_1-11](https://user-images.githubusercontent.com/20401990/156698829-412f73f9-f0c6-46bb-8231-3bf7e2585942.png)

Same vectors / different viewing agles:
![Figure_2](https://user-images.githubusercontent.com/20401990/156698830-7dee9cd0-385b-480c-b733-ac8006e24802.png)
![Figure_1](https://user-images.githubusercontent.com/20401990/156698831-4b180a1a-6719-4fbf-972a-440a3710f252.png)






