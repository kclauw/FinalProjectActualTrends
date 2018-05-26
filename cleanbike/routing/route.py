from urllib.parse import urlencode
from requests import get
import numpy as np
import pandas as pd
from random import randint
from time import sleep
from util import append_to_2dlist


KEY = 'AIzaSyC8ezA75-Lhm4C6GWowFDrH9fLI19AqtOU'


#https://maps.googleapis.com/maps/api/distancematrix/json?key=AIzaSyC8ezA75-Lhm4C6GWowFDrH9fLI19AqtOU&
def _fetch(api, args):

    base_url = 'https://maps.googleapis.com/maps/api/%s/json?key=%s&'
    return get(base_url % (api, KEY) + urlencode(args)).json()

def _decode(polyline):

    values, current = [], []
    for byte in bytearray(polyline, 'ascii'):

        byte -= 63

        current.append(byte & 0x1f)
        if byte & 0x20:

            continue

        value = 0
        for chunk in reversed(current):

            value <<= 5
            value |= chunk

        values.append(((~value if value & 0x1 else value) >> 1) / 100000.)
        current = []

    result, x, y = [], 0., 0.
    for dx, dy in [tuple(values[i:i+2]) for i in range(0, len(values), 2)]:

        x, y = x + dx, y + dy
        result.append((round(x, 6), round(y, 6)))

    return result



def coordinates(origins,destination):
    '''
    Get the coordinates of a given location.
    Returns None if it didn't work.
    '''

    try:

        result = _fetch('geocode', {'address': location})
        result = result['results'][0]['geometry']['location']

    except:

        return None

    return result['lat'], result['lng']


def distanceMatrix(origins, destinations = None, mode = 'walking'):
    '''
    Find a not-so-long path from somewhere to somewhere else.
    'origin' and 'destination' must be coordinates.
    If 'destination' is omitted, 'origin' must be a list of coordinates.
        >>> path(coordinates('Paris'), coordinates('London'))
        [(48.85668, 2.35196), (48.86142, 2.33929), (48.86897, 2.32369), ...
    Returns a list of coordinates to follow, or None if it didn't work either.
    '''
    
    args = {'mode': mode}
    #Convert the origins and destinations to a format the google API understands .i.e  long1,lat1|long2,lat2
    args.update({'origins' : "|".join(str(",".join(str(x) for x in y)) for y in origins), 'destinations' : "|".join(str(",".join(str(x) for x in y)) for y in destinations)})
    try:

        result = _fetch('distancematrix', args)
        # You can extract other useful informations here, such as duration
        #print(result['routes'][0])
        distance_matrix=[]
        for isrc, source in enumerate(result['origin_addresses']):
            temp_row=[]
            for idst, target in enumerate(result['destination_addresses']):
                row = result['rows'][isrc]
                cell = row['elements'][idst]
                #if cell['status'] == 'OK':
                 #   print('{} to {}: {}, {}.'.format(source, target, cell['distance']['text'], cell['duration']['text']))
                #else:
                #    print('{} to {}: status = {}'.format(source, target, cell['status']))
                temp_row.append((cell['distance']['text'],cell['duration']['text']))
            distance_matrix.append(temp_row)
            #print(temp_row)
        return distance_matrix
    except:
        print("except")
        return None

    return distance_matrix



def name(lat,lng):
    '''
    Get the name of a given location.
    According to the geocoordinates
    Returns None if it didn't work.
    '''

    try:



        result = _fetch('geocode', {'lat': lat,'lng': lng})
        result = result['results'][0]
        print("r",result)

    except:

        return None

    return result['formatted_address']


def distance(lat,lng):
    '''
    Get the distance between to points.
    According to the geocoordinates
    Returns None if it didn't work.
    '''

    try:



        result = _fetch('geocode', {'lat': lat,'lng': lng})
        result = result['results'][0]
        print("r",result)

    except:

        return None

    return result['formatted_address']




def path(origin, destination = None, mode = 'walking'):
    '''
    Find a not-so-long path from somewhere to somewhere else.
    'origin' and 'destination' must be coordinates.
    If 'destination' is omitted, 'origin' must be a list of coordinates.
        >>> path(coordinates('Paris'), coordinates('London'))
        [(48.85668, 2.35196), (48.86142, 2.33929), (48.86897, 2.32369), ...
    Returns a list of coordinates to follow, or None if it didn't work either.
    '''

    args = {'mode': mode}
    if destination:

        origin, destination = '%f,%f' % origin, '%f,%f' % destination

    else:

        points = ['%f,%f' % coordinates for coordinates in origin]
        origin, destination = points[0], points[-1]

        if len(points) > 2:

            args = {'waypoints': 'optimize:true|via:' + '|via:'.join(points[1:-1])}

    args.update({'origin': origin, 'destination': destination})
    try:

        result = _fetch('directions', args)
        # You can extract other useful informations here, such as duration
        #print(result['routes'][0])
        result = result['routes'][0]['overview_polyline']['points']
       

    except:

        return None

    return _decode(result)

def distance_duration(origin, destination = None, mode = 'walking'):
    '''
    Find a not-so-long path from somewhere to somewhere else.
    'origin' and 'destination' must be coordinates.
    If 'destination' is omitted, 'origin' must be a list of coordinates.
        >>> path(coordinates('Paris'), coordinates('London'))
        [(48.85668, 2.35196), (48.86142, 2.33929), (48.86897, 2.32369), ...
    Returns a list of coordinates to follow, or None if it didn't work either.
    '''

    args = {'mode': mode}
    sleep(1)
    if destination:

        origin, destination = '%f,%f' % origin, '%f,%f' % destination

    else:

        points = ['%f,%f' % coordinates for coordinates in origin]
        origin, destination = points[0], points[-1]

        if len(points) > 2:

            args = {'waypoints': 'optimize:true|via:' + '|via:'.join(points[1:-1])}

    args.update({'origin': origin, 'destination': destination, 'units' : 'metric'})
    try:

        result = _fetch('directions', args)
        # You can extract other useful informations here, such as duration
        #print(result['routes'][0])
        distance = result['routes'][0]['legs'][0]['distance']['value']
        duration = result['routes'][0]['legs'][0]['duration']['value']
        #result = result['routes'][0]['overview_polyline']['points']
       

    except:

        return None

    return distance, duration




#print(coordinates('Paris'))
origins = tuple([50.824352000000005,4.334465])

#origins = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]

#print(add_chunk_to_matrix(origins, 10))
destinations = tuple([50.87808,4.34793])
L = [1,2,3]       
print(path(origins,destinations))
#Process a batch

#WARNING: this is far from clean code (but it works..)
def batch_distance_matrix(x,batch, mode = 'walking'):
    length = len(x)
    distance_matrix=[]
    i = 0
    distance_matrix = [[] for i in range(length)]

    while i < length:
       
        #Process a batch of data
        chunk1 = x[i:(i+batch)]
        print("i",i,len(chunk1),len(distance_matrix[i]))
        #Compare chunk to other elements in dataset and create a row
        j = 0
        row = []
        
        while j < length:

            chunk2 = x[j:(j+batch)]

            #dm = distanceMatrix(chunk1,chunk2)

            #args = {'mode': mode}

            #Convert the origins and destinations to a format the google API understands .i.e  long1,lat1|long2,lat2
            #args.update({'origins' : "|".join(str(",".join(str(x) for x in y)) for y in chunk1), 'destinations' : "|".join(str(",".join(str(x) for x in y)) for y in chunk2)})
            #result = _fetch('distancematrix', args)
            

            for source in range(len(chunk1)):
                print("source",source+i,len(distance_matrix))
                #distance_matrix[j]
                for target in range(len(chunk2)):
                    #print("target",target)

                    #print("target",i+source," len ",len(distance_matrix))
                    distance_matrix[i+source].append(target)
                #for target in range(len(chunk2)):
                    
                    #distance_matrix[i+source].append(source)
            # You can extract other useful informations here, such as duration
            
            #for isrc, source in enumerate(result['origin_addresses']):
            #    for idst, target in enumerate(result['destination_addresses']):
            #        row = result['rows'][isrc]
            #        cell = row['elements'][idst]
             #       distance_matrix[i+isrc].append((cell['distance']['text'],cell['duration']['text']))
           
            


            #sleep(1)
            j+=batch

            #row = append_to_2dlist(dm, row)

        #print("after batch ",distance_matrix)
        #print(distance_matrix)
        df = pd.DataFrame(distance_matrix,index=None)
        df.to_csv('./data/batch' + str(i) + '.csv',index=None)
        
        i+=batch

    #Process the rest
    return distance_matrix




    




