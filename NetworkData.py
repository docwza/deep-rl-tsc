import sys, subprocess, os
import inspect

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    #from sumolib import checkBinary
    import sumolib
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

###now can import SUMO traci module
import traci
import numpy as np

class NetworkData():
    def __init__(self, net_fp):
        print(net_fp) 
        self.net = sumolib.net.readNet(net_fp)
        ###get edge data
        self.edge_data = self.get_edge_data(self.net)
        self.lane_data = self.get_lane_data(self.net)
        self.intersection_data = self.get_intersection_data(self.net)
        print("SUCCESSFULLY GENERATED NET DATA")

    def get_net_data(self):
        return {'lane':self.lane_data, 'edge':self.edge_data, 'origins':self.find_origin_edges(), 'destinations':self.find_destination_edges()}

    def get_intersection_data(self, net):
        nodes = [n for n in net.getNodes()]

        nodes = net.getNodes()
        node_data = {str(node.getID()):{} for node in nodes} 
                                                                                          
        for node in nodes:
            node_ID = str(node.getID())
            node_data[node_ID]['incoming'] = [str(lane.getID()) for lane in node.getIncoming()]
            node_data[node_ID]['outgoing'] = [str(lane.getID()) for lane in node.getOutgoing()]

    def find_destination_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['incoming']:
                next_edges[next_e] += 1
                                                                 
        destinations = [ e for e in next_edges if next_edges[e] == 0]
        return destinations

    def find_origin_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['outgoing']:
                next_edges[next_e] += 1

        origins = [ e for e in next_edges if next_edges[e] == 0]
        return origins

    def get_edge_data(self, net):
        edges = net.getEdges()
        edge_data = {str(edge.getID()):{} for edge in edges} 

        for edge in edges:
            edge_ID = str(edge.getID())
            edge_data[edge_ID]['lanes'] = [str(lane.getID()) for lane in edge.getLanes()]
            edge_data[edge_ID]['length'] = float(edge.getLength())
            edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getOutgoing()]
            edge_data[edge_ID]['noutgoing'] = len(edge_data[edge_ID]['outgoing'])
            edge_data[edge_ID]['nlanes'] = len(edge_data[edge_ID]['lanes'])
            edge_data[edge_ID]['incoming'] = [str(inc.getID()) for inc in edge.getIncoming()]
            edge_data[edge_ID]['outnode'] = str(edge.getFromNode().getID())
            edge_data[edge_ID]['incnode'] = str(edge.getToNode().getID())
            edge_data[edge_ID]['speed'] = float(edge.getSpeed())
          
            ###coords for each edge
            incnode_coord = edge.getFromNode().getCoord()
            outnode_coord = edge.getToNode().getCoord()
            edge_data[edge_ID]['coord'] = np.array([incnode_coord[0], incnode_coord[1], outnode_coord[0], outnode_coord[1]]).reshape(2,2)
            #print edge_data[edge_ID]['coord']
        return edge_data 

    def get_lane_data(self, net):
        #create lane objects from lane_ids
        lane_ids = []
        for edge in self.edge_data:
            lane_ids.extend(self.edge_data[edge]['lanes'])

        lanes = [net.getLane(lane) for lane in lane_ids]
        #lane data dict
        lane_data = {lane:{} for lane in lane_ids}

        for lane in lanes:
            lane_id = lane.getID()
            lane_data[ lane_id ]['length'] = lane.getLength()
            lane_data[ lane_id ]['edge'] = str(lane.getEdge().getID())
            #lane_data[ lane_id ]['outgoing'] = []
            lane_data[ lane_id ]['outgoing'] = {}
            ###.getOutgoing() returns a Connection type, which we use to determine the next lane
            moveid = []
            for conn in lane.getOutgoing():
                #lane_data[ lane_id ]['outgoing'] = str(conn.getToLane().getID())
                out_id = str(conn.getToLane().getID())
                lane_data[ lane_id ]['outgoing'][out_id] = {'dir':str(conn.getDirection()), 'index':conn.getTLLinkIndex()}
                moveid.append(str(conn.getDirection()))
                ###ADD CODE HERE FOR CONNECTION TYPE
            lane_data[ lane_id ]['movement'] = ''.join(sorted(moveid))
            #create empty list for incoming lanes 
            lane_data[ lane_id ]['incoming'] = []
               
        #determine incoming lanes using outgoing lanes data
        for lane in lane_data:
            for inc in lane_data:
                if lane == inc:
                    continue
                else:
                    if inc in lane_data[lane]['outgoing']:
                        lane_data[inc]['incoming'].append(lane)

        return lane_data
