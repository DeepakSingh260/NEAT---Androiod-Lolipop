import pygame 
import os
import random
import time
import neat
import visualize
import pickle
pygame.init()
pygame.font.init()

STAT_FONT = pygame.font.SysFont("comicsans",50)
WIN = pygame.display.set_mode((600,800))
Android_image = pygame.transform.scale(pygame.image.load('Android.png'),(64,64))
base_image = pygame.transform.scale(pygame.image.load("background.png"),(600,800)).convert_alpha()
Lolipop_image = pygame.transform.scale2x(pygame.image.load("Lolipop.png")).convert_alpha()
gen =0
class Android:
	img = Android_image
	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.vel = 0
		self.ticks = 0
		self.height =self.y


	def jump(self):
		self.ticks = 0
		self.vel=-10	
		self.height=self.y

	def move(self):
		self.ticks+=1
		displacement = self.vel*self.ticks + (1.5)*(self.ticks)**2

		if displacement >16:
			displacement =16

		if displacement < 0:
			displacement-=2

		self.y =self.y +displacement	

	def draw(self,win):
			win.blit(Android_image,(self.x,self.y))


	def get_mask(self):
		return pygame.mask.from_surface(Android_image)		


def window_draw(win , Androids,lolipops,score,gen,pipe_ind):
	if gen == 0:
		gen=1
	bg = win.blit(base_image,(0,0))
	for lolipop in lolipops:
		lolipop.draw(win)

	for android in Androids:

		android.draw(win)
	score_label = STAT_FONT.render('Score'+str(score),1,(255,255,255))
	win.blit(score_label,(600-score_label.get_width()-15,10))
	gen_label = STAT_FONT.render('Gens:'+str(gen-1),1,(255,255,255))
	win.blit(gen_label,(10,10))
	alive_label = STAT_FONT.render('alive'+str(len(Androids)),1,(255,255,255))
	win.blit(alive_label,(10,50))
	
	

	pygame.display.update()	

class Lolipop:
	GAP = 300
	
	def __init__(self,x):
		self.x = x
		self.top=0
		self.bottom = 0
		self.height = 0
		self.LOLIPOP_BOTTOM = Lolipop_image
		self.LOLIPOP_TOP = pygame.transform.flip(Lolipop_image,False,True)
		self.passed =False

		self.set_height()

	def set_height(self):
		self.height= random.randrange(50,450)
		self.top = self.height - self.LOLIPOP_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x-=5

	def draw(self,win):
		win.blit(self.LOLIPOP_TOP,(self.x,self.top))
		win.blit(self.LOLIPOP_BOTTOM,(self.x,self.bottom))



	def collision(self,android,win):
		android_mask = android.get_mask()
		top_mask = pygame.mask.from_surface(self.LOLIPOP_TOP)
		bottom_mask = pygame.mask.from_surface(self.LOLIPOP_BOTTOM)

		top_offset= (self.x -android.x,self.top - round(android.y))	
		bottom_offset= (self.x -android.x,self.bottom - round(android.y))	
			
		b_point = android_mask.overlap(bottom_mask,bottom_offset)
		t_point = android_mask.overlap(top_mask,top_offset)

		if t_point or b_point:
			return True

		return False	
def fitness_function(genomes,config):
	global WIN, gen
	win = WIN
	gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
	networks = []
	androids = []
	gene = []
	for genome_id, genome in genomes:
		genome.fitness = 0  # start with fitness level of 0
		network = neat.nn.FeedForwardNetwork.create(genome, config)
		networks.append(network)
		androids.append(Android(230,350))
		gene.append(genome)


	lolipops = [Lolipop(700)]
	score = 0

	clock = pygame.time.Clock()

	run = True
	while run and len(androids) > 0:
		clock.tick(30)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				break

		lolipop_ind = 0
		if len(androids) > 0:
			if len(lolipops) > 1 and androids[0].x > lolipops[0].x + lolipops[0].LOLIPOP_TOP.get_width():  # determine whether to use the first or second
				lolipop_ind = 1                                                                 # pipe on the screen for neural network input

		for x, android in enumerate(androids):  # give each bird a fitness of 0.1 for each frame it stays alive
			gene[x].fitness += 0.1
			android.move()

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
			output = networks[androids.index(android)].activate((android.y, abs(android.y - lolipops[lolipop_ind].height), abs(android.y - lolipops[lolipop_ind].bottom)))

			if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
				android.jump()



		rem = []
		add_lolipop = False
		for lolipop in lolipops:
			lolipop.move()
			# check for collision
			for android in androids:
				if lolipop.collision(android, win):
					gene[androids.index(android)].fitness -= 1
					networks.pop(androids.index(android))
					gene.pop(androids.index(android))
					androids.pop(androids.index(android))


				if not lolipop.passed and lolipop.x < android.x:
					lolipop.passed = True
					add_lolipop = True


			if lolipop.x + lolipop.LOLIPOP_TOP.get_width() < 0:
				rem.append(lolipop)    

		if add_lolipop:
			score += 1
            
			for genome in gene:
				genome.fitness += 5
			lolipops.append(Lolipop(700))

		for r in rem:
			lolipops.remove(r)

		for android in androids:
			if android.y + android.img.get_height() - 10 >= 800 or android.y < -50:
				networks.pop(androids.index(android))
				gene.pop(androids.index(android))
				androids.pop(androids.index(android))

		window_draw(WIN, androids, lolipops,  score, gen, lolipop_ind)

def run(config_file):
	config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,
		neat.DefaultStagnation,config_file)
	population =neat.Population(config)
	population.add_reporter(neat.StdOutReporter(True))
	Statisitics = neat.StatisticsReporter()
	population.add_reporter(Statisitics)
	Best_Gene = population.run(fitness_function,50)



if __name__ == '__main__':
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir,'config-feedforward.txt')
	run(config_path)


