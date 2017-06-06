from cluster import *



if __name__=='__main__':
	for i in range(1,10):
		points = readPoints('result/a'+str(i)+'.txt')
		deleteOutPoints(points , 'result/b'+str(i)+'.txt')
		k,centers,lable = multiCluster([4,10],points,3)
		
		potentials = potentialEnergy(points , k , centers , lable)
		originPoints = readPoints('result/b'+str(i)+'.txt')
		filler(originPoints , centers , potentials)
		outputFile('filled/lable'+str(i)+'.txt', originPoints)
		print(i , centers)