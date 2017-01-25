# -*- coding: utf-8 -*-

import constants as CONST

class Road():
    
    def reInit(self):
        self.cells = []
        temp = ([False] * CONST.CELLS_PER_LANE)
        for i in range(len(CONST.LANES) - 1):
            self.cells.append(temp[:])
    
    def __inti__(self):
        self.reInit()
    
    def __col(self, loc):
        return loc[0] // CONST.CAR_LENGTH

    def __row(self, loc):
        return loc[1] // CONST.LANE_WIDTH
            
    def setCell(self, loc):
        
        self.cells[self.__col(loc)][self.__row(loc)] = True        

    def clearCell(self, loc):
        self.cells[self.__col(loc)][self.__row(loc)] = False
            
    def isEmpty(self, loc):
        return self.cells[self.__col(loc)][self.__row(loc)]
        
    
