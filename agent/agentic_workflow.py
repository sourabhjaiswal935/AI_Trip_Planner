from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool


class GraphBuilder():
    def __init__(self,model_provider:str="groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        self.tools=[
            # WeatherInfoTool(),
            # PlaceSearchTool(),
            # CalculatorTool(),
            # CurrencyConverterTool()
        ]

        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()

        self.tools.extend([* self.weather_tools.weather_tool_list, 
                           * self.place_search_tools.place_search_tool_list,
                           * self.calculator_tools.calculator_tool_list,
                           * self.currency_converter_tools.currency_converter_tool_list])
        
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        
        self.graph = None
        
        self.system_prompt=SYSTEM_PROMPT

    def agent_function(self, state:MessagesState):
        """Main agent Function it make decision to call tools or not based on the user question and system prompt"""
        user_question=state["messages"]
        input_question=[self.system_prompt] + user_question
        response=self.llm_with_tools.invoke(input_question)
        return {"messages" : [response]}


    def build_graph(self):
        # state=It stores the information that nodes need to read or update.
        # It is the data structure used to carry information through the graph execution. Persistence can be added separately.
        graph_builder=StateGraph(MessagesState)
        #a node can read the info from state and update the state with new info. The state is passed to the node when it is executed.
        #A node is a Python function that performs a specific task in the workflow.
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START,"agent")
        graph_builder.add_conditional_edges("agent",tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)
        
        self.graph = graph_builder.compile()
        return self.graph

    def __call__(self):
        return self.build_graph()